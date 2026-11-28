//! Media-reference forwarding: decide per request whether the router
//! preprocesses media itself or forwards references to a gRPC worker that
//! advertises worker-side processing, and build the wire payload.

use std::{
    borrow::Cow,
    collections::{BTreeSet, HashSet},
    sync::Arc,
};

use axum::response::Response;
use llm_multimodal::{
    data_url::{base64_payload, data_url_payload, data_url_payload_bytes},
    media, MediaConnectorError, MediaContentPart, MediaErrorCode, Modality,
};
use openai_protocol::worker::MmProcessingMode;
use smg_grpc_client::{common_proto as common, vllm_proto as vllm};
use smg_worker::{
    identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType},
    labels::SUPPORTS_VISION_LABEL,
};

use super::{
    capability::{runtime_supports_modality, worker_takes_stripped_refs_side},
    config::MultimodalComponents,
    plan::{MediaPlan, PlaceholderTokens},
};
use crate::{
    observability::metrics::Metrics,
    routers::{
        common::{
            placement::{self, PipelinePin, NO_MEDIA_REF_CAPABLE_WORKER_CODE},
            snapshot::WorkerRegistry,
        },
        error,
        grpc::{
            common::stages::pd_protocol::{EncodeHandoff, PdProtocol},
            context::WorkerSelection,
        },
    },
};

/// Worker label carrying the advertised backend (`GetServerInfo.mm_processor`).
pub(crate) const MM_PROCESSOR_LABEL: &str = "mm_processor";
/// Worker label carrying accepted URL schemes (`GetServerInfo.mm_media_ref_schemes`).
pub(crate) const MM_MEDIA_REF_SCHEMES_LABEL: &str = "mm_media_ref_schemes";
/// The `mm_processor` value of a worker that runs the gateway's own media
/// pipeline in place of the engine's processor. It expands a family's anchor
/// exactly as the gateway would, so a spec's opt-in for the engine's prompt
/// updates (`worker_expandable`) does not bind it; it takes image and video
/// references, never audio.
pub(crate) const GATEWAY_PIPELINE_PROCESSOR: &str = "smg";

/// Where one request's media is fetched and preprocessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MmProcessing {
    Router,
    Worker,
}

impl MmProcessing {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Router => "router",
            Self::Worker => "worker",
        }
    }
}

/// Why a request cannot be forwarded as media references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MmRefsError {
    /// A part carries per-item processing hints (the long-side cap, the
    /// sampling rate) no worker honors, and the router cannot apply them
    /// itself: it has no spec for the model, or the request carries a
    /// modality only the worker processes (audio).
    HintUnsupported,
    /// The model's placeholder anchor is not one the engine's own processor
    /// expands, and the model's workers do not all run the gateway's pipeline.
    ModelNotOptedIn(Modality),
    /// The selected worker runs the engine's own processor, which does not expand
    /// this model's anchor; only a worker running the gateway's pipeline does.
    PipelineWorkerRequired,
    /// The selected worker runs the gateway's own pipeline, which takes no audio.
    PipelineTakesNoAudio,
    /// vLLM does not accept this modality at all.
    ModalityUnsupported(Modality),
    /// A part kind with no reference form (inline bytes or embeddings).
    UnsupportedPart(&'static str),
    /// A reference the request got wrong, refused at the door before any
    /// worker is selected, in the media connector's own words (`cause`): a
    /// url that is empty or begins with whitespace or a control character, a
    /// data URL whose header names no base64 payload or whose payload is
    /// empty, an inline part whose media type is not one `type/subtype` pair
    /// or whose payload itself opens with a data URL header. The request's
    /// own fault, under the code the router-side pipeline answers invalid
    /// media with (`invalid_multimodal_request`); the media type read has no
    /// router-side counterpart (see `check_forwarded_references`).
    InvalidReference { index: usize, cause: String },
    /// An inline `data:` payload above the router's byte cap.
    RefTooLarge {
        index: usize,
        bytes: usize,
        limit: usize,
    },
    /// The selected worker does not fetch this URL scheme.
    SchemeNotAccepted { scheme: String, accepted: String },
    /// EPD encode workers never process references.
    EncodeNotSupported,
    /// A selected worker does not advertise worker-side processing.
    WorkerNotCapable,
    /// The model has no registry spec, so the router cannot process its media
    /// itself, and the placement did not give it a worker that does.
    RouterSpecMissing,
    /// The model has registered workers and none of them processes media
    /// references, under the placement that forwards every request's media
    /// to the worker (`--mm-processing worker`). A verdict over the fleet,
    /// not a fault of the request: a terminal 503 ([`Self::into_response`]).
    NoCapableWorker { model: String, workers: usize },
}

impl MmRefsError {
    /// Stable error code for the HTTP response body.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::HintUnsupported => "multimodal_hint_unsupported_in_worker_mode",
            Self::ModelNotOptedIn(_) | Self::PipelineWorkerRequired => {
                "multimodal_worker_processing_unsupported_model"
            }
            Self::ModalityUnsupported(_)
            | Self::UnsupportedPart(_)
            | Self::PipelineTakesNoAudio => "multimodal_not_supported",
            Self::InvalidReference { .. } => MediaErrorCode::InvalidRequest.as_str(),
            Self::RefTooLarge { .. } => "media_ref_too_large",
            Self::SchemeNotAccepted { .. } => "media_ref_scheme_not_accepted",
            Self::EncodeNotSupported | Self::WorkerNotCapable | Self::RouterSpecMissing => {
                "multimodal_not_supported"
            }
            Self::NoCapableWorker { .. } => NO_MEDIA_REF_CAPABLE_WORKER_CODE,
        }
    }

    /// The client's response: a fault of the request is a 400 under the
    /// variant's code; the fleet verdict ([`Self::NoCapableWorker`]) is the
    /// 503 worker selection has always answered that condition with
    /// ([`placement::no_media_ref_capable_worker`]), terminal for the retry
    /// layer and reported once, rate-limited, like the pool's other refusals.
    pub(crate) fn into_response(self) -> Response {
        let code = self.code();
        let message = self.to_string();
        match self {
            Self::NoCapableWorker { .. } => placement::no_media_ref_capable_worker(message),
            _ => error::bad_request(code, message),
        }
    }
}

impl std::fmt::Display for MmRefsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HintUnsupported => f.write_str(
                "per-item media hints (max_long_side_pixel, fps) cannot be forwarded to a worker, \
                 and the router cannot apply them itself: it has no media spec for this model, \
                 or an item is of a modality only the worker processes",
            ),
            Self::ModelNotOptedIn(modality) => write!(
                f,
                "this model's {modality} placeholder is not expandable by a vLLM worker running \
                 the engine's own processor; workers running the gateway's own pipeline \
                 (mm_processor={GATEWAY_PIPELINE_PROCESSOR}) expand it, when every \
                 media-processing worker of the model is one"
            ),
            Self::PipelineWorkerRequired => write!(
                f,
                "the selected worker runs the engine's own processor, which does not expand this \
                 model's placeholder; only a worker running the gateway's own pipeline \
                 (mm_processor={GATEWAY_PIPELINE_PROCESSOR}) takes this model's media"
            ),
            Self::PipelineTakesNoAudio => write!(
                f,
                "the selected worker runs the gateway's own pipeline \
                 (mm_processor={GATEWAY_PIPELINE_PROCESSOR}), which takes image and video \
                 references, not audio; a worker with --mm-processor inprocess or redis \
                 processes audio with the engine's own processor"
            ),
            Self::ModalityUnsupported(modality) => {
                write!(f, "vLLM workers do not accept {modality} inputs")
            }
            Self::UnsupportedPart(kind) => {
                write!(f, "{kind} cannot be forwarded as a media reference")
            }
            Self::InvalidReference { index, cause } => write!(f, "media item {index}: {cause}"),
            Self::RefTooLarge {
                index,
                bytes,
                limit,
            } => write!(
                f,
                "media item {index}: inline payload is {bytes} bytes, above the {limit}-byte cap"
            ),
            Self::SchemeNotAccepted { scheme, accepted } => {
                write!(
                    f,
                    "the selected worker does not fetch '{scheme}' URLs (accepts: {accepted})"
                )?;
                // The remedy is the refused scheme's own: a file URL needs the
                // worker's local-media flag; any other scheme, the empty one of
                // a bare path or a port-less host written without a scheme
                // included, has none to name beyond the accepted list.
                if scheme == "file" {
                    f.write_str("; file:// needs --allowed-local-media-path on the worker")?;
                }
                Ok(())
            }
            Self::EncodeNotSupported => f.write_str(
                "media references cannot be routed through encode workers (encode-prefill-decode \
                 pipeline); run the router with --mm-processing auto or router",
            ),
            Self::WorkerNotCapable => f.write_str(
                "the selected worker does not advertise worker-side multimodal processing",
            ),
            Self::RouterSpecMissing => f.write_str(
                "the router has no media spec for this model's family and cannot process its \
                 media itself; a vLLM worker with --mm-processor inprocess|redis processes it \
                 under --mm-processing worker (or auto, when every worker of the model does and \
                 none is an encode worker: media references never travel on the \
                 encode-prefill-decode pipeline, so that pipeline serves no media the router \
                 cannot process)",
            ),
            Self::NoCapableWorker { model, workers } => write!(
                f,
                "model '{model}' has {workers} registered worker(s) and none processes media \
                 references (no worker advertises an mm_processor), so under --mm-processing \
                 worker its requests that forward media references cannot be served: start the \
                 workers with worker-side media processing, or run the router with \
                 --mm-processing auto or router"
            ),
        }
    }
}

/// Whether `worker` can fetch and process media references itself.
pub(crate) fn worker_accepts_media_refs(worker: &dyn Worker) -> bool {
    let spec = &worker.metadata().spec;
    *worker.connection_mode() == ConnectionMode::Grpc
        && spec
            .labels
            .get(MM_PROCESSOR_LABEL)
            .is_some_and(|value| match spec.runtime_type {
                RuntimeType::Vllm => !value.is_empty(),
                RuntimeType::Sglang => value
                    .trim()
                    .eq_ignore_ascii_case(GATEWAY_PIPELINE_PROCESSOR),
                _ => false,
            })
}

pub(crate) fn worker_supports_ec_transfer(worker: &dyn Worker) -> bool {
    worker
        .metadata()
        .spec
        .labels
        .get("supports_ec_transfer")
        .is_some_and(|value| value == "true")
}

/// Whether `worker` processes media references with the gateway's own
/// pipeline ([`GATEWAY_PIPELINE_PROCESSOR`]) rather than the engine's.
pub(crate) fn worker_runs_gateway_pipeline(worker: &dyn Worker) -> bool {
    worker_accepts_media_refs(worker)
        && worker
            .metadata()
            .spec
            .labels
            .get(MM_PROCESSOR_LABEL)
            .is_some_and(|value| {
                value
                    .trim()
                    .eq_ignore_ascii_case(GATEWAY_PIPELINE_PROCESSOR)
            })
}

/// Whether the plan carries a modality whose anchor only the gateway's own
/// pipeline expands: the model has a spec, and that modality is not opted in
/// for the engine's prompt updates. Such a request is placed only on workers
/// that run the pipeline ([`worker_runs_gateway_pipeline`]); a model without
/// a spec renders the engine's own placeholders and needs no pin.
fn plan_needs_gateway_pipeline(plan: &MediaPlan, placeholders: &PlaceholderTokens) -> bool {
    !placeholders.is_worker_only()
        && plan
            .modalities()
            .iter()
            .any(|&modality| !placeholders.worker_expandable(modality))
}

/// Whether the model's workers that process media references all run the
/// gateway's own pipeline (and there is at least one): then a family without
/// the engine-side opt-in forwards to them.
fn fleet_runs_gateway_pipeline(registry: &WorkerRegistry, model_id: &str) -> bool {
    let workers = registry.get_by_model(model_id);
    let mut capable = workers
        .iter()
        .filter(|worker| worker_accepts_media_refs(worker.as_ref()))
        .peekable();
    capable.peek().is_some() && capable.all(|worker| worker_runs_gateway_pipeline(worker.as_ref()))
}

/// Whether `worker` runs an engine with a vision encoder (`supports_vision`
/// `"true"`, see [`SUPPORTS_VISION_LABEL`]) on the gRPC pipeline and does not
/// process media references: under worker-side placement no image or video
/// ever reaches it. A worker without the label is unknown, never this.
fn vision_worker_without_media_processing(worker: &dyn Worker) -> bool {
    worker.connection_mode().uses_grpc_pipeline()
        && worker
            .metadata()
            .spec
            .labels
            .get(SUPPORTS_VISION_LABEL)
            .is_some_and(|value| value == "true")
        && !worker_accepts_media_refs(worker)
}

/// Whether `workers` hold an encode worker, so their model runs the
/// encode-prefill-decode pipeline: media references never travel on it,
/// whatever its workers advertise. Under `--mm-processing auto` such a pool
/// resolves to router-side processing; under `worker`, selection refuses the
/// references with the request's own 400 (`EncodeNotSupported`), which names
/// the fix that applies there (`--mm-processing auto` or `router`). Such a
/// model is neither refused in preparation nor named as waiting for a
/// media-capable worker (readiness, the registration warning): a capable
/// worker changes nothing for it.
fn pool_has_encode_worker<'a>(workers: impl IntoIterator<Item = &'a Arc<dyn Worker>>) -> bool {
    workers
        .into_iter()
        .any(|worker| matches!(worker.worker_type(), WorkerType::Encode))
}

/// Whether `pool` holds a worker that processes media references
/// ([`worker_accepts_media_refs`]): under worker-side placement the model's
/// forwarded media has a worker to go to, and selection picks it.
pub(crate) fn pool_processes_media_refs<'a>(
    pool: impl IntoIterator<Item = &'a Arc<dyn Worker>>,
) -> bool {
    pool.into_iter()
        .any(|worker| worker_accepts_media_refs(worker.as_ref()))
}

/// Whether registering `worker` under worker-side placement is said in the
/// log: a vision worker that does not process media references
/// ([`vision_worker_without_media_processing`]) of a model that is not on the
/// encode-prefill-decode pipeline (`pool`: the model's registered workers and
/// the ones registering with it, see [`pool_has_encode_worker`]).
pub(crate) fn registration_lacks_media_processing<'a>(
    worker: &dyn Worker,
    pool: impl IntoIterator<Item = &'a Arc<dyn Worker>>,
) -> bool {
    vision_worker_without_media_processing(worker) && !pool_has_encode_worker(pool)
}

/// The models among `workers` whose requests that forward media references
/// worker-side placement cannot serve: a model with a worker that reports a
/// vision encoder and no worker that processes media references, judged over
/// every registered worker (healthy or not: the capability is configuration).
/// What `/readiness` names in its ready body, sorted for a stable line and
/// body. Empty under any other placement: `auto` processes such a model's media
/// on the router, `router` always does. A model with an encode worker is never
/// named: on the encode-prefill-decode pipeline references are refused whatever
/// its workers advertise ([`pool_has_encode_worker`]), so a capable worker
/// would change nothing for it.
pub(crate) fn models_without_media_processing(
    processing: MmProcessingMode,
    workers: &[Arc<dyn Worker>],
) -> Vec<String> {
    if processing != MmProcessingMode::Worker {
        return Vec::new();
    }
    let mut vision_only = BTreeSet::new();
    let mut capable = HashSet::new();
    let mut encode_prefill_decode = HashSet::new();
    for worker in workers {
        if matches!(worker.worker_type(), WorkerType::Encode) {
            encode_prefill_decode.insert(worker.model_id());
        } else if worker_accepts_media_refs(worker.as_ref()) {
            capable.insert(worker.model_id());
        } else if vision_worker_without_media_processing(worker.as_ref()) {
            vision_only.insert(worker.model_id());
        }
    }
    vision_only
        .into_iter()
        .filter(|model| !capable.contains(model) && !encode_prefill_decode.contains(model))
        .map(str::to_string)
        .collect()
}

/// URL schemes `worker` advertised; empty when it advertised none.
fn worker_media_ref_schemes(worker: &dyn Worker) -> HashSet<String> {
    worker
        .metadata()
        .spec
        .labels
        .get(MM_MEDIA_REF_SCHEMES_LABEL)
        .map(|value| {
            value
                .split(',')
                .map(|scheme| scheme.trim().to_ascii_lowercase())
                .filter(|scheme| !scheme.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Whether a worker can expand every anchor of `plan`: the engine's own
/// processor expands the modalities the spec opted in; a fleet whose
/// media-processing workers all run the gateway's own pipeline
/// (`pipeline_fleet`) expands any spec's image or video anchor, as the
/// gateway would, and no audio anchor at all (the pipeline takes no audio).
fn ensure_worker_expandable(
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
    pipeline_fleet: bool,
) -> Result<(), MmRefsError> {
    for &modality in plan.modalities() {
        // Audio uses the native worker processor; preprocessed audio tensors
        // remain unsupported on vLLM's separate router-processing path.
        if modality != Modality::Audio && !runtime_supports_modality(RuntimeType::Vllm, modality) {
            return Err(MmRefsError::ModalityUnsupported(modality));
        }
        let pipeline_expands = pipeline_fleet && modality != Modality::Audio;
        if !placeholders.worker_expandable(modality) && !pipeline_expands {
            return Err(MmRefsError::ModelNotOptedIn(modality));
        }
    }
    Ok(())
}

/// Which of the workers that process references may take a forwarded plan:
/// only those running the gateway's own pipeline when an anchor needs it
/// ([`plan_needs_gateway_pipeline`]), never those when the plan carries
/// audio (the pipeline takes none), either otherwise. Set in preparation,
/// read by selection and by the retry pin.
pub(crate) fn media_refs_pipeline_pin(
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
) -> PipelinePin {
    if plan_needs_gateway_pipeline(plan, placeholders) {
        PipelinePin::Only
    } else if plan.modalities().contains(&Modality::Audio) {
        PipelinePin::Never
    } else {
        PipelinePin::Any
    }
}

/// Whether the router's own pipeline can serve `plan` for a vLLM worker: it has
/// a spec for the model, and every modality is one it preprocesses for that
/// runtime (audio is the worker's own processor's alone). Checked before the
/// hint fallback, so a request the pipeline would refuse at worker selection
/// keeps the cheap refusal instead of a fetch and a decode first.
fn router_serves(plan: &MediaPlan, placeholders: &PlaceholderTokens) -> bool {
    !placeholders.is_worker_only()
        && plan
            .modalities()
            .iter()
            .all(|&modality| runtime_supports_modality(RuntimeType::Vllm, modality))
}

/// Under worker-side placement, a model whose registered workers include none
/// that processes media references cannot serve a forwarded request whichever
/// worker selection would pick: refused here, in preparation, before the
/// request is weighed against any worker and ahead of the selection stage's
/// own refusal of the condition (its backstop for a pool that changed in
/// between). Every registered worker counts, healthy or not, as under `auto`:
/// the capability is configuration, not health. A model with no worker at all
/// is left to selection, which answers for it (the 404 of a model nobody
/// serves, or the disaggregated side that is missing). So is a pool with an
/// encode worker, the encode-prefill-decode pipeline: media references never
/// travel on it whatever its workers advertise, and selection refuses them
/// with the request's own 400 (`EncodeNotSupported`), which names the fix
/// that applies there (`--mm-processing auto` or `router`, not a worker-side
/// processor).
fn ensure_fleet_processes_media_refs(
    registry: &WorkerRegistry,
    model_id: &str,
) -> Result<(), MmRefsError> {
    let workers = registry.get_by_model(model_id);
    if workers.is_empty()
        || pool_has_encode_worker(workers.iter())
        || pool_processes_media_refs(workers.iter())
    {
        return Ok(());
    }
    Err(MmRefsError::NoCapableWorker {
        model: model_id.to_string(),
        workers: workers.len(),
    })
}

/// Decide where this request's media is processed. Runs in preparation, before
/// worker selection, so `auto` consults every registered worker of the model,
/// and `worker` refuses a model none of whose workers can take the request
/// ([`ensure_fleet_processes_media_refs`]).
pub(crate) fn resolve_mm_processing(
    components: &MultimodalComponents,
    registry: &WorkerRegistry,
    model_id: &str,
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
) -> Result<MmProcessing, MmRefsError> {
    let (resolved, reason) = match components.processing {
        MmProcessingMode::Router => (MmProcessing::Router, "config"),
        MmProcessingMode::Worker => match plan_carries_hints(plan)? {
            // The hints are knobs of the router's own pipeline (the long-side
            // cap, the sampling rate): a hinted request that pipeline can serve
            // is preprocessed here, as under `auto`; every other request keeps
            // forwarding references.
            true if router_serves(plan, placeholders) => (MmProcessing::Router, "hint_fallback"),
            true => return Err(MmRefsError::HintUnsupported),
            false => {
                // A family the engine's processor does not expand still
                // forwards when every media-processing worker of the model
                // runs the gateway's own pipeline; selection then places the
                // request on those workers only.
                let pipeline_fleet = plan_needs_gateway_pipeline(plan, placeholders)
                    && fleet_runs_gateway_pipeline(registry, model_id);
                ensure_worker_expandable(plan, placeholders, pipeline_fleet)?;
                ensure_fleet_processes_media_refs(registry, model_id)?;
                (
                    MmProcessing::Worker,
                    if pipeline_fleet {
                        "pipeline_workers"
                    } else {
                        "config"
                    },
                )
            }
        },
        MmProcessingMode::Auto => {
            if plan.is_forwardable() {
                // Every registered worker, healthy or not: a health flap must
                // not flip a model between modes request to request.
                let workers = registry.get_by_model(model_id);
                // An encode pool whose encoders take references under a
                // push handoff processes them itself; its prefill/decode
                // workers consume the identity and need no media processor.
                let epd = pool_has_encode_worker(workers.iter());
                let push_encode = epd
                    && workers
                        .iter()
                        .filter(|worker| *worker.worker_type() == WorkerType::Encode)
                        .all(|worker| {
                            PdProtocol::encode_handoff(worker.metadata().spec.runtime_type)
                                == Some(EncodeHandoff::Push)
                                && worker_supports_ec_transfer(worker.as_ref())
                                && worker_accepts_media_refs(worker.as_ref())
                        });
                let uniform = push_encode
                    || (!workers.is_empty()
                        && !epd
                        && workers
                            .iter()
                            .all(|worker| worker_accepts_media_refs(worker.as_ref())));
                // The gateway's own pipeline on every worker expands any
                // spec's anchor; a fleet mixing it with the engine's
                // processors keeps the spec's opt-in.
                let pipeline_fleet = plan_needs_gateway_pipeline(plan, placeholders)
                    && uniform
                    && workers
                        .iter()
                        .all(|worker| worker_runs_gateway_pipeline(worker.as_ref()));
                if ensure_worker_expandable(plan, placeholders, pipeline_fleet).is_err() {
                    (MmProcessing::Router, "model_not_opted_in")
                } else if workers.is_empty() {
                    (MmProcessing::Router, "auto_none")
                } else if uniform {
                    (
                        MmProcessing::Worker,
                        if pipeline_fleet {
                            "auto_pipeline"
                        } else {
                            "auto_uniform"
                        },
                    )
                } else if epd {
                    (MmProcessing::Router, "auto_epd")
                } else {
                    (MmProcessing::Router, "auto_mixed")
                }
            } else {
                (MmProcessing::Router, "plan_not_forwardable")
            }
        }
    };

    if resolved == MmProcessing::Worker {
        // The references a worker would be handed, read once at the door:
        // a shape no engine serves is refused here, in the connector's
        // words, before a worker is selected.
        check_forwarded_references(plan)?;
    }

    if resolved == MmProcessing::Router && placeholders.is_worker_only() {
        // No spec: the router has nothing to process the media with.
        return Err(MmRefsError::RouterSpecMissing);
    }

    // Where a model's media is processed is worker configuration, reported in
    // the worker's labels and countable on this metric. It is not a per-request
    // event, so it is not logged as one.
    Metrics::record_mm_processing(
        registry.metric_model_label(model_id),
        resolved.as_str(),
        reason,
    );
    Ok(resolved)
}

/// Post-selection check: every worker the references can reach must accept
/// them, and every URL scheme in the plan must be one every such worker that
/// advertised schemes fetches (the prefill worker processes them; the decode
/// worker does too whenever the identity relay does not apply, as with n>1,
/// but never when `refs_stripped_decode` and the decode worker is a vLLM
/// language-model-only one: that side is served the prefill's media identity,
/// stripped to text and content hashes, so no reference reaches it). A plan
/// pinned to the gateway's own pipeline (`pin`, see
/// [`media_refs_pipeline_pin`]) needs every worker on that pipeline, and a
/// worker on it takes no audio; selection draws its candidates by the same
/// pin, so this is the backstop.
pub(crate) fn ensure_selection_supports_media_refs(
    workers: &WorkerSelection,
    plan: &MediaPlan,
    pin: PipelinePin,
    refs_stripped_decode: bool,
) -> Result<(), MmRefsError> {
    let needs_pipeline = pin == PipelinePin::Only;
    let workers: Vec<&dyn Worker> = match workers {
        WorkerSelection::Single { worker } => vec![&**worker],
        WorkerSelection::Disaggregated {
            encode_assignments: Some(assignments),
            ..
        } if !assignments.is_empty() => assignments
            .iter()
            .map(|assignment| assignment.worker.as_ref())
            .collect(),
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => {
            if refs_stripped_decode && worker_takes_stripped_refs_side(&**decode) {
                vec![prefill.as_ref()]
            } else {
                vec![prefill.as_ref(), &**decode]
            }
        }
    };
    if !workers
        .iter()
        .all(|worker| worker_accepts_media_refs(*worker))
    {
        return Err(MmRefsError::WorkerNotCapable);
    }
    let carries_audio = plan.modalities().contains(&Modality::Audio);
    for worker in &workers {
        let pipeline = worker_runs_gateway_pipeline(*worker);
        if needs_pipeline && !pipeline {
            return Err(MmRefsError::PipelineWorkerRequired);
        }
        if pipeline && carries_audio {
            return Err(MmRefsError::PipelineTakesNoAudio);
        }
    }
    let mut accepted: Option<HashSet<String>> = None;
    for worker in &workers {
        let schemes = worker_media_ref_schemes(*worker);
        if schemes.is_empty() {
            continue;
        }
        accepted = Some(match accepted {
            Some(current) => current.intersection(&schemes).cloned().collect(),
            None => schemes,
        });
    }
    let Some(accepted) = accepted else {
        return Ok(());
    };
    for part in plan.parts() {
        let Some(scheme) = part_scheme(part) else {
            continue;
        };
        if !accepted.contains(&scheme) {
            let mut accepted: Vec<_> = accepted.into_iter().collect();
            accepted.sort();
            return Err(MmRefsError::SchemeNotAccepted {
                scheme,
                accepted: accepted.join(","),
            });
        }
    }
    Ok(())
}

/// The scheme a part's reference is fetched by: a URL part's own, `data`
/// for a base64 source.
fn part_scheme(part: &MediaContentPart) -> Option<String> {
    match part {
        MediaContentPart::ImageUrl { url, .. }
        | MediaContentPart::VideoUrl { url, .. }
        | MediaContentPart::AudioUrl { url, .. } => Some(url_scheme(url)),
        MediaContentPart::ImageBase64 { .. } | MediaContentPart::AudioBase64 { .. } => {
            Some("data".to_string())
        }
        _ => None,
    }
}

/// The data URL a base64 source stands for, built for a wire that takes URLs.
fn data_url(media_type: &str, data: &str) -> String {
    format!("data:{media_type};base64,{data}")
}

/// The wire form of a forwardable part's URL: the text as the request wrote
/// it, or the base64 source a data URL is built from. Classifying a part
/// names the form; only a wire that takes the URL builds it.
enum WireUrl<'a> {
    AsWritten(&'a str),
    Base64 { media_type: &'a str, data: &'a str },
}

impl<'a> WireUrl<'a> {
    /// The URL the wire takes: borrowed as written, or the data URL built
    /// once from the base64 source.
    fn build(&self) -> Cow<'a, str> {
        match self {
            Self::AsWritten(url) => Cow::Borrowed(url),
            Self::Base64 { media_type, data } => Cow::Owned(data_url(media_type, data)),
        }
    }
}

/// Lower-cased URL scheme; `data` for data URLs, empty for bare paths and for
/// a url that begins with whitespace: the reference is forwarded as written and
/// the byte caps read the scheme at offset zero, so nothing reads past it here
/// (the door refuses such a url as an invalid URL).
fn url_scheme(url: &str) -> String {
    url.split_once(':')
        .map(|(scheme, _)| scheme)
        .filter(|scheme| {
            !scheme.is_empty()
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        })
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

/// Classify one part as a forwardable reference: its modality, the wire form
/// of its URL (built only where a wire takes it) and the router's byte cap
/// for inline payloads.
fn forwardable_ref(
    part: &MediaContentPart,
) -> Result<(common::Modality, WireUrl<'_>, usize), MmRefsError> {
    match part {
        MediaContentPart::ImageUrl {
            url,
            max_long_side_pixel: None,
            ..
        } => Ok((
            common::Modality::Image,
            WireUrl::AsWritten(url.as_str()),
            media::image_max_input_bytes(),
        )),
        MediaContentPart::ImageBase64 {
            media_type,
            data,
            max_long_side_pixel: None,
            ..
        } => Ok((
            common::Modality::Image,
            WireUrl::Base64 { media_type, data },
            media::image_max_input_bytes(),
        )),
        MediaContentPart::VideoUrl {
            url,
            fps: None,
            max_long_side_pixel: None,
            ..
        } => Ok((
            common::Modality::Video,
            WireUrl::AsWritten(url.as_str()),
            media::video_max_input_bytes(),
        )),
        MediaContentPart::ImageUrl { .. }
        | MediaContentPart::ImageBase64 { .. }
        | MediaContentPart::VideoUrl { .. } => Err(MmRefsError::HintUnsupported),
        MediaContentPart::ImageData { .. } => {
            Err(MmRefsError::UnsupportedPart("inline image bytes"))
        }
        MediaContentPart::VideoData { .. } => {
            Err(MmRefsError::UnsupportedPart("inline video bytes"))
        }
        MediaContentPart::AudioUrl { url, .. } => Ok((
            common::Modality::Audio,
            WireUrl::AsWritten(url.as_str()),
            media::audio_max_input_bytes(),
        )),
        MediaContentPart::AudioBase64 {
            media_type, data, ..
        } => Ok((
            common::Modality::Audio,
            WireUrl::Base64 { media_type, data },
            media::audio_max_input_bytes(),
        )),
        MediaContentPart::AudioData { .. } => {
            Err(MmRefsError::UnsupportedPart("inline audio bytes"))
        }
        MediaContentPart::ImageEmbeds { .. } => {
            Err(MmRefsError::UnsupportedPart("image embeddings"))
        }
        MediaContentPart::Text { .. } => Err(MmRefsError::UnsupportedPart("text")),
    }
}

/// Whether this plan is forwardable apart from per-item hints: `Ok(true)` when
/// a part carries one, `Ok(false)` when none does; a part with no reference
/// form at all is the error, whatever its position. Classification only: no
/// part's URL is built here.
fn plan_carries_hints(plan: &MediaPlan) -> Result<bool, MmRefsError> {
    let mut hinted = false;
    for part in plan.parts() {
        match forwardable_ref(part) {
            Ok(_) => {}
            Err(MmRefsError::HintUnsupported) => hinted = true,
            Err(blocker) => return Err(blocker),
        }
    }
    Ok(hinted)
}

/// The characters of a refused url its refusal echoes, at most: the sentence
/// names the url the client sent, never a whole inline payload.
const URL_ECHO_CHARS: usize = 64;

/// The first `URL_ECHO_CHARS` characters of a refused url, an ellipsis when cut.
fn echoed_url(url: &str) -> String {
    let mut echoed: String = url.chars().take(URL_ECHO_CHARS).collect();
    if url.chars().nth(URL_ECHO_CHARS).is_some() {
        echoed.push_str("...");
    }
    echoed
}

/// The references a forwarded plan carries, read once at the door before any
/// worker is selected, by the rules the router-side pipeline's fetch applies
/// to the same media: a url that is empty or begins with whitespace or a
/// control character (the reference is forwarded as written and the byte caps
/// read the scheme at offset zero; the refusal echoes at most the first
/// `URL_ECHO_CHARS` characters of it), a data URL whose header names no base64
/// payload and an empty payload are refused here in the connector's own
/// words, instead of being forwarded to a worker no engine of which serves
/// them. An inline part's header is read before the door builds its data URL
/// (one `type/subtype` pair of the registration grammar's characters; a
/// payload that does not itself open with a data URL header), since the
/// request names the media type and the wire takes the URL as built. That
/// read is this door's own, with no router-side counterpart: the router
/// placement decodes a base64 source's payload alone and never its media
/// type, so under the automatic placement a malformed header is answered by
/// the placement the request lands on. The symbols and the length of a
/// base64 payload stay the engine's verdict: nothing here reads the payload
/// past the five bytes a data URL header would occupy.
fn check_forwarded_references(plan: &MediaPlan) -> Result<(), MmRefsError> {
    let refused = |index: usize, cause: MediaConnectorError| MmRefsError::InvalidReference {
        index,
        cause: cause.to_string(),
    };
    for (index, part) in plan.parts().iter().enumerate() {
        match part {
            MediaContentPart::ImageUrl { url, .. }
            | MediaContentPart::VideoUrl { url, .. }
            | MediaContentPart::AudioUrl { url, .. } => {
                // The reference is forwarded as written and every byte cap reads
                // the scheme at offset zero: nothing is trimmed here either, so a
                // url that is empty or begins with a control character (DEL and
                // the C1 range among them) or any whitespace is an invalid URL,
                // echoed by a bounded prefix. No scheme begins with one.
                if url.is_empty() || url.starts_with(|c: char| c.is_control() || c.is_whitespace())
                {
                    return Err(refused(
                        index,
                        MediaConnectorError::InvalidUrl(echoed_url(url)),
                    ));
                }
                if opens_a_data_url(url) {
                    data_url_payload(url).map_err(|cause| refused(index, cause))?;
                }
            }
            MediaContentPart::ImageBase64 {
                media_type, data, ..
            }
            | MediaContentPart::AudioBase64 {
                media_type, data, ..
            } => {
                // The wire takes a data URL built here from the part's media
                // type and payload (`data_url`); the header's shape is read
                // before it is built, the payload's symbols are not. The chat
                // door writes `audio/<format>` from `input_audio.format` as
                // sent, so a MIME type or a stray character there is a media
                // type that is not one type/subtype pair; a payload that
                // itself opens with a data URL header (a data URL sent where
                // bare base64 is documented; read trimmed, as the payload rule
                // reads it) would be wrapped a second time. Either is refused
                // as the invalid URL it would make, echoed from the payload's
                // head alone: a refusal copies no payload whole.
                if !media_type_is_a_type_pair(media_type) || opens_a_data_url(data.trim_start()) {
                    let payload_head: String = data.chars().take(URL_ECHO_CHARS).collect();
                    return Err(refused(
                        index,
                        MediaConnectorError::InvalidUrl(echoed_url(&data_url(
                            media_type,
                            &payload_head,
                        ))),
                    ));
                }
                base64_payload(data).map_err(|cause| refused(index, cause))?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Whether `media_type` is one `type/subtype` pair of the media type
/// registration grammar's restricted-name characters (RFC 6838 section 4.2:
/// an alphanumeric first, then alphanumerics and `!#$&-^_.+`), the shape a
/// data URL header carries before its parameters: both halves present,
/// neither holding a second slash, a parameter separator or whitespace.
fn media_type_is_a_type_pair(media_type: &str) -> bool {
    let Some((kind, subtype)) = media_type.split_once('/') else {
        return false;
    };
    let restricted_name = |part: &str| {
        part.starts_with(|c: char| c.is_ascii_alphanumeric())
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "!#$&-^_.+".contains(c))
    };
    restricted_name(kind) && restricted_name(subtype)
}

/// Whether `text` opens with a data URL header (`data:` in any case): a URL
/// part's url that is a data URL, or a base64 payload where the request sent
/// a data URL in place of the bare base64 documented.
fn opens_a_data_url(text: &str) -> bool {
    text.get(..5)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
}

/// Build the wire payload from the plan, in authored (prompt) order.
pub(crate) fn assemble_media_refs(plan: &MediaPlan) -> Result<vllm::MediaRefs, MmRefsError> {
    let mut items = Vec::with_capacity(plan.parts().len());
    for (index, part) in plan.parts().iter().enumerate() {
        let (modality, url, limit) = forwardable_ref(part)?;
        let url = url.build();
        if let Some(bytes) = data_url_payload_bytes(&url) {
            if bytes > limit {
                return Err(MmRefsError::RefTooLarge {
                    index,
                    bytes,
                    limit,
                });
            }
        }
        items.push(vllm::MediaRef {
            modality: modality as i32,
            url: url.into_owned(),
        });
    }
    Ok(vllm::MediaRefs { items })
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/refs/tests.rs"]
mod tests;
