//! What a request demands of a prefill/decode pair beyond the pool and the
//! wire, judged over the pair index before the placement picks the roles:
//! which sides need a vision encoder, whether the workers must take the
//! request's forwarded media references, and the context window the prompt
//! must fit. One judge for the gRPC selection stage and the HTTP
//! prefill-decode router; each builds the demand from what its path knows of
//! the request (the gRPC stage from its decoded media and tokenized prompt,
//! the HTTP router from the typed body's media parts and token ids).

use axum::response::Response;
use tracing::warn;

use super::placement::{CandidateFilter, PairFailure, PipelinePin};
use crate::{
    routers::{
        error,
        grpc::{
            common::stages::{pair_context_length, ContextRequirements},
            context::WireConstraint,
            multimodal,
        },
    },
    worker::{PdPairIndex, PrefillCandidateError, Worker, WorkerType},
};

/// Why a prefill/decode selection produced nothing: the placement's own
/// verdict for the role that failed, a request carrying pixel media for a
/// model that pairs no prefill worker with a vision encoder, or a prompt no
/// pair's context window holds (the 400 itself).
#[derive(Debug)]
pub(crate) enum PairSelectionFailure {
    Placement(Box<PairFailure>),
    NoVisionCapablePrefill,
    Context(Box<Response>),
}

impl From<Box<PairFailure>> for PairSelectionFailure {
    fn from(failure: Box<PairFailure>) -> Self {
        Self::Placement(failure)
    }
}

impl PrefillCandidateError for PairSelectionFailure {
    fn is_at_capacity(&self) -> bool {
        matches!(self, Self::Placement(failure) if failure.is_at_capacity())
    }
}

/// Candidate predicate for requests carrying media references.
pub(crate) fn accepts_media_refs(worker: &dyn Worker) -> bool {
    multimodal::worker_accepts_media_refs(worker)
}

/// Candidate predicate for a request, or a disaggregated side of one, that
/// needs a vision encoder: a worker that advertises none
/// (`supports_vision=false`) admits such a request and never schedules it. A
/// worker without the label reads as capable, as today.
pub(crate) fn vision_capable(worker: &dyn Worker) -> bool {
    !multimodal::worker_language_model_only(worker)
}

/// The predicate of a request whose prefill side needs a vision encoder, on
/// either side: a prefill worker with the encoder, any decode worker.
pub(crate) fn prefill_vision_capable(worker: &dyn Worker) -> bool {
    !matches!(worker.worker_type(), WorkerType::Prefill) || vision_capable(worker)
}

/// Whether `worker` is on the side of `pin`: running the gateway's own
/// pipeline for [`PipelinePin::Only`], not running it for
/// [`PipelinePin::Never`], anything for [`PipelinePin::Any`].
pub(crate) fn pipeline_admits(pin: PipelinePin, worker: &dyn Worker) -> bool {
    match pin {
        PipelinePin::Any => true,
        PipelinePin::Only => multimodal::worker_runs_gateway_pipeline(worker),
        PipelinePin::Never => !multimodal::worker_runs_gateway_pipeline(worker),
    }
}

/// Whether `pairs` holds a prefill that `accepts` admits with a partner it
/// admits too, both on `wire` when a retry pins one. Read from the index,
/// not the pools, so a worker that pairs with nothing cannot shut the pool.
pub(crate) fn pair_admitted_by(
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    accepts: &dyn Fn(&dyn Worker) -> bool,
) -> bool {
    let admitted = |worker: &std::sync::Arc<dyn Worker>| {
        WireConstraint::admits(wire, worker.as_ref()) && accepts(worker.as_ref())
    };
    pairs
        .prefill
        .iter()
        .zip(&pairs.partners)
        .filter(|(prefill, _)| admitted(prefill))
        .any(|(prefill, partners)| {
            let runtime = prefill.metadata().spec.runtime_type;
            partners
                .iter()
                .any(|decode| admitted(decode) && decode.metadata().spec.runtime_type == runtime)
        })
}

/// The 400 for `workers` (the regular pool's, or the prefill pool's) that
/// serve the model while none has a vision encoder, so none can take the
/// request's images or video: the 400 a model without vision gives image
/// input, instead of a dispatch the engine admits and never schedules. A
/// 400 is terminal for the retry layer by status. `function` and `mode`
/// name the selection site the line is written from.
pub(crate) fn no_vision_capable_workers(
    function: &'static str,
    mode: &'static str,
    model_id: &str,
    workers: &str,
) -> Response {
    warn!(
        function,
        mode,
        model_id = %model_id,
        workers,
        "No worker that accepts vision input for a request carrying pixel media"
    );
    error::bad_request(
        "multimodal_not_supported",
        format!(
            "image or video input is not supported by model '{model_id}': none of its \
             {workers} accept vision input (the engine runs as a language model only, or \
             its servicer does not carry media to it)"
        ),
    )
}

/// Context bounds for both token-processing sides, and which sides need a vision
/// encoder. The prefill side takes the request's media whole (the processed
/// pixel tensors, or the references it expands itself), so any image or
/// video is a demand on it: a prefill worker without the encoder admits
/// such a request and never schedules it, and no dispatch check catches
/// that. The decode side's demand is the caller's (the gRPC stage's
/// `decode_needs_vision`; on the HTTP path every pixel request's, whose
/// body each side expands itself), and holds on the prefill side too
/// (neither side runs those shapes without the encoder).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PairDemands {
    pub(crate) context: ContextRequirements,
    pub(crate) prefill: bool,
    pub(crate) decode: bool,
    /// The request forwards media references, which only a worker that
    /// processes them can take: the prefill side's refusal check then wants
    /// a worker with the encoder among those, or the request could never
    /// pair and a retryable 503 would send the client back for ever.
    pub(crate) refs: bool,
    /// With `refs`: which of the workers that process them the plan may use
    /// ([`pipeline_admits`]); the probes count only the pairs on that side,
    /// so a pair the pin excludes never stands in for one it admits.
    pub(crate) pipeline: PipelinePin,
}

impl PairDemands {
    /// The demand of a request that carries pixel media (`pixel_media`, see
    /// `multimodal::request_needs_vision`) and whose decode side needs the
    /// encoder (`decode`).
    pub(crate) fn new(pixel_media: bool, decode: bool) -> Self {
        Self {
            context: ContextRequirements::default(),
            prefill: pixel_media || decode,
            decode,
            refs: false,
            pipeline: PipelinePin::Any,
        }
    }

    /// The same demand for a request that forwards media references (`refs`,
    /// the caller's references predicate) on the side of `pipeline`: the
    /// prefill side's refusal and the capacity refusal are judged among the
    /// workers that take them there.
    pub(crate) fn with_media_refs(self, refs: bool, pipeline: PipelinePin) -> Self {
        Self {
            refs,
            pipeline: if refs { pipeline } else { PipelinePin::Any },
            ..self
        }
    }

    /// Context bounds shared by both token-processing sides, never the encoder.
    pub(crate) fn with_context(self, context: ContextRequirements) -> Self {
        Self { context, ..self }
    }

    /// The demand on the side `worker` serves.
    pub(crate) fn on(self, worker: &dyn Worker) -> bool {
        match worker.worker_type() {
            WorkerType::Prefill => self.prefill,
            _ => self.decode,
        }
    }

    /// Judge `pairs` by these demands ahead of the placement, inside
    /// `caller`'s own candidate predicate when it has one; the result is the
    /// predicate the placement selects under ([`PairAdmission::fits`]).
    ///
    /// A side that needs a vision encoder is kept off the workers without
    /// one. The prefill side must find one: a request carrying pixel media
    /// is refused when the index pairs no prefill worker with the encoder (a
    /// role nobody serves and a fleet without a shared transport keep the
    /// placement's own verdicts), since no dispatch check would catch the
    /// side landing on a worker without it. The decode side is kept off such
    /// workers while the index holds a pair with the encoder on both sides
    /// that the caller's own predicate admits too; with none the decode pool
    /// is used as it is, and dispatch refuses the side naming the gap, as it
    /// does for a pool with no such worker at all. With media references the
    /// worker must take them too (`refs`): a pool whose referencing prefill
    /// worker lacks the encoder and whose encoder-bearing one takes no
    /// references can never pair the request, and the verdict of the
    /// unfiltered pool would be a retryable 503 over healthy workers. A pool
    /// with no pair that takes the references at all is not this refusal:
    /// the placement's media-references verdict names that gap, and the fix
    /// that applies to it.
    ///
    /// A prompt no admitted pair's context window holds is refused with the
    /// largest window among them; membership, not health, decides whether a
    /// request can ever fit, and the caller's attempt-local exclusions do
    /// not count (a failed large worker does not make the client's prompt
    /// oversized).
    pub(crate) fn judge<'a>(
        self,
        pairs: &PdPairIndex,
        wire: Option<WireConstraint>,
        caller: Option<CandidateFilter<'a>>,
        model_id: &'a str,
    ) -> Result<PairAdmission<'a>, PairSelectionFailure> {
        let takes_refs = |worker: &dyn Worker| {
            !self.refs || (accepts_media_refs(worker) && pipeline_admits(self.pipeline, worker))
        };
        if self.prefill
            && !pairs.prefill_pool.is_empty()
            && !pairs.decode_pool.is_empty()
            && pairs.refusal.is_none()
            && pair_admitted_by(pairs, wire, &takes_refs)
            && !pair_admitted_by(pairs, wire, &|worker: &dyn Worker| {
                prefill_vision_capable(worker) && takes_refs(worker)
            })
        {
            return Err(PairSelectionFailure::NoVisionCapablePrefill);
        }
        let admitted = move |worker: &dyn Worker| caller.is_none_or(|accepts| accepts(worker));
        let with_vision = move |worker: &dyn Worker| {
            (!self.on(worker) || vision_capable(worker)) && admitted(worker)
        };
        let admission = PairAdmission {
            demand: self,
            caller,
            require_decode_vision: self.decode && pair_admitted_by(pairs, wire, &with_vision),
            model_id,
        };
        // Inspect actual partners: independently large pools need not pair.
        let takes_media = |worker: &dyn Worker| admission.takes_media(worker);
        if !pair_admitted_by(pairs, wire, &|worker: &dyn Worker| {
            takes_media(worker) && self.context.admits(worker, model_id)
        }) {
            let mut largest = None;
            for (prefill, partners) in pairs.prefill.iter().zip(&pairs.partners) {
                if !WireConstraint::admits(wire, prefill.as_ref()) || !takes_media(prefill.as_ref())
                {
                    continue;
                }
                for decode in partners.iter().filter(|decode| {
                    WireConstraint::admits(wire, decode.as_ref())
                        && takes_media(decode.as_ref())
                        && decode.metadata().spec.runtime_type
                            == prefill.metadata().spec.runtime_type
                }) {
                    let limit = pair_context_length(prefill.as_ref(), decode.as_ref(), model_id);
                    largest = largest.max(limit);
                }
            }
            if let Some(limit) = largest {
                self.context
                    .enforce(limit, model_id)
                    .map_err(|response| PairSelectionFailure::Context(Box::new(response)))?;
            }
        }
        Ok(admission)
    }
}

/// The pair index as a request's demands admit it ([`PairDemands::judge`]):
/// the predicate the placement selects under.
pub(crate) struct PairAdmission<'a> {
    demand: PairDemands,
    caller: Option<CandidateFilter<'a>>,
    /// The decode side is kept off the workers without the encoder: the
    /// index holds a pair with the encoder on both sides that the caller
    /// admits.
    require_decode_vision: bool,
    model_id: &'a str,
}

impl PairAdmission<'_> {
    /// The vision predicate the placement selects under: with the decode
    /// demand met somewhere, each side needs the encoder it demands; else a
    /// pixel request's prefill side alone; a text request has none.
    fn vision_admits(&self, worker: &dyn Worker) -> bool {
        if self.require_decode_vision {
            !self.demand.on(worker) || vision_capable(worker)
        } else if self.demand.prefill {
            prefill_vision_capable(worker)
        } else {
            true
        }
    }

    /// The media demands alone, for the capacity refusal's view of the
    /// index: the caller's attempt-local exclusions do not count there.
    fn takes_media(&self, worker: &dyn Worker) -> bool {
        let demand = self.demand;
        (!demand.refs || (accepts_media_refs(worker) && pipeline_admits(demand.pipeline, worker)))
            && (!demand.prefill || prefill_vision_capable(worker))
            && (!self.require_decode_vision || !demand.on(worker) || vision_capable(worker))
    }

    /// Whether the placement may select `worker`: the caller's predicate,
    /// the side's vision demand and the context window.
    pub(crate) fn fits(&self, worker: &dyn Worker) -> bool {
        self.caller.is_none_or(|accepts| accepts(worker))
            && self.vision_admits(worker)
            && self.demand.context.admits(worker, self.model_id)
    }
}
