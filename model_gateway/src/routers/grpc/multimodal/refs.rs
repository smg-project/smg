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
    capability::runtime_supports_modality,
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

/// Post-selection check: every worker must accept references, and every URL
/// scheme in the plan must be one every worker that advertised schemes fetches
/// (the prefill worker processes them; the decode worker does too whenever the
/// identity relay does not apply, as with n>1). A plan pinned to the
/// gateway's own pipeline (`pin`, see [`media_refs_pipeline_pin`]) needs
/// every worker on that pipeline, and a worker on it takes no audio; selection
/// draws its candidates by the same pin, so this is the backstop.
pub(crate) fn ensure_selection_supports_media_refs(
    workers: &WorkerSelection,
    plan: &MediaPlan,
    pin: PipelinePin,
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
        } => vec![prefill.as_ref(), &**decode],
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
mod tests {
    use std::sync::Arc;

    use openai_protocol::model_card::ModelCard;
    use smg_worker::{identity::basic::WorkerType, test_support::no_health_check};

    use super::*;
    #[cfg(all(
        feature = "jemalloc-stats",
        not(target_env = "msvc"),
        not(target_env = "musl")
    ))]
    use crate::routers::grpc::testing::bytes_allocated_during;
    use crate::routers::{
        common::snapshot::test_support::registry_with,
        grpc::{
            backend_client::BasicWorkerBuilder,
            multimodal::{config::MultimodalConfigRegistry, settings::MultimodalSettings},
        },
    };

    const MODEL: &str = "qwen3-vl";

    fn image_url(url: &str) -> MediaContentPart {
        MediaContentPart::ImageUrl {
            url: url.to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }
    }

    fn plan(parts: Vec<MediaContentPart>) -> MediaPlan {
        MediaPlan::new(parts)
    }

    fn base64_image(data: &str) -> MediaContentPart {
        MediaContentPart::ImageBase64 {
            media_type: "image/png".to_string(),
            data: Arc::new(data.to_string()),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }
    }

    fn base64_audio(data: &str) -> MediaContentPart {
        MediaContentPart::AudioBase64 {
            media_type: "audio/wav".to_string(),
            data: Arc::new(data.to_string()),
            uuid: None,
        }
    }

    fn expandable_placeholders() -> PlaceholderTokens {
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, "<|image_pad|>".to_string());
        placeholders.set_worker_expandable(Modality::Image, true);
        placeholders
    }

    fn components(mode: MmProcessingMode) -> MultimodalComponents {
        let mut components = MultimodalComponents::new(
            Arc::new(MultimodalConfigRegistry::new()),
            None,
            None,
            &MultimodalSettings::default(),
        )
        .expect("components");
        components.processing = mode;
        assert!(!components.drain_fetches_on_error);
        components
    }

    fn worker(
        url: &str,
        runtime: RuntimeType,
        connection: ConnectionMode,
        labels: &[(&str, &str)],
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(MODEL))
            .worker_type(WorkerType::Regular)
            .runtime_type(runtime)
            .connection_mode(connection)
            .health_config(no_health_check());
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        Arc::new(builder.build())
    }

    fn capable(url: &str) -> Arc<dyn Worker> {
        worker(
            url,
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[
                (MM_PROCESSOR_LABEL, "inprocess"),
                (MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data"),
            ],
        )
    }

    type AcceptCase = (
        RuntimeType,
        ConnectionMode,
        &'static [(&'static str, &'static str)],
        bool,
    );

    #[test]
    /// Forward references only to gRPC workers advertising a supported processor.
    fn accepts_only_supported_labeled_grpc_workers() {
        let cases: [AcceptCase; 8] = [
            (
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "inprocess")],
                true,
            ),
            (
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "")],
                false,
            ),
            (RuntimeType::Vllm, ConnectionMode::Grpc, &[], false),
            (
                RuntimeType::Vllm,
                ConnectionMode::Zmq,
                &[(MM_PROCESSOR_LABEL, "inprocess")],
                false,
            ),
            (
                RuntimeType::Sglang,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "inprocess")],
                false,
            ),
            (
                RuntimeType::Sglang,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "smg")],
                true,
            ),
            (RuntimeType::Sglang, ConnectionMode::Grpc, &[], false),
            (
                RuntimeType::Sglang,
                ConnectionMode::Zmq,
                &[(MM_PROCESSOR_LABEL, "smg")],
                false,
            ),
        ];
        for (i, (runtime, connection, labels, expected)) in cases.into_iter().enumerate() {
            let scheme = if connection == ConnectionMode::Zmq {
                format!("ipc:///tmp/smg-refs-{i}.ipc")
            } else {
                format!("grpc://127.0.0.1:{}", 9100 + i)
            };
            let w = worker(&scheme, runtime, connection, labels);
            assert_eq!(worker_accepts_media_refs(w.as_ref()), expected, "case {i}");
        }
    }

    /// An inline clip or image whose data URL header the request got wrong
    /// is refused at the door as the invalid URL it would make, echoed by a
    /// bounded prefix: a media type that is not one type/subtype pair (the
    /// chat door writes `audio/<format>` from `input_audio.format` as sent, so
    /// a MIME type there doubles the header and an empty format breaks it),
    /// or a payload that itself opens with a data URL header. A restricted-name
    /// subtype of any case passes; the payload's symbols stay unread.
    #[test]
    fn an_inline_payload_whose_header_the_request_got_wrong_is_refused_at_the_door() {
        fn clip(media_type: &str, data: &str) -> MediaContentPart {
            MediaContentPart::AudioBase64 {
                media_type: media_type.to_string(),
                data: data.to_string().into(),
                uuid: None,
            }
        }
        fn image(media_type: &str, data: &str) -> MediaContentPart {
            MediaContentPart::ImageBase64 {
                media_type: media_type.to_string(),
                data: data.to_string().into(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            }
        }
        let refused = |part: MediaContentPart, url: &str| {
            assert_eq!(
                check_forwarded_references(&plan(vec![part])),
                Err(MmRefsError::InvalidReference {
                    index: 0,
                    cause: format!("invalid media URL: {url}"),
                }),
                "{url}"
            );
        };
        // A MIME type as the format: the header the door would build doubles it.
        refused(
            clip("audio/audio/wav", "UklG"),
            "data:audio/audio/wav;base64,UklG",
        );
        // An empty format: no subtype; a subtype opening with a separator.
        refused(clip("audio/", "UklG"), "data:audio/;base64,UklG");
        refused(clip("audio/+x", "UklG"), "data:audio/+x;base64,UklG");
        // No slash (the Messages door copies `source.media_type` as sent), a
        // parameter, a trailing space: not one type/subtype pair.
        refused(image("jpeg", "AAAA"), "data:jpeg;base64,AAAA");
        refused(
            clip("audio/wav;rate=16000", "UklG"),
            "data:audio/wav;rate=16000;base64,UklG",
        );
        refused(image("image/png ", "AAAA"), "data:image/png ;base64,AAAA");
        // A data URL sent as the payload (read trimmed): wrapped a second time.
        refused(
            clip("audio/wav", " data:audio/wav;base64,UklG"),
            "data:audio/wav;base64, data:audio/wav;base64,UklG",
        );
        refused(
            image("image/png", "DATA:image/png;base64,AAAA"),
            "data:image/png;base64,DATA:image/png;base64,AAAA",
        );
        // The echo is bounded.
        let long = "A".repeat(200);
        let Err(MmRefsError::InvalidReference { cause, .. }) =
            check_forwarded_references(&plan(vec![clip("audio/x/y", &long)]))
        else {
            panic!("a doubled header is refused");
        };
        assert_eq!(
            cause.len(),
            "invalid media URL: ".len() + URL_ECHO_CHARS + 3
        );
        assert!(cause.ends_with("..."));
        // Well-formed headers pass whatever the payload's symbols or the subtype's case.
        for part in [
            clip("audio/wav", "UklG"),
            clip("audio/WAV", "UklG"),
            clip("audio/x-wav", "UklG"),
            clip("audio/xyz", "not base64 at all"),
            image("image/svg+xml", "PHN2Zz4="),
        ] {
            check_forwarded_references(&plan(vec![part])).expect("a well-formed header passes");
        }
    }

    /// A reference no engine serves is refused at the door, before a worker
    /// is selected, in the media connector's own words under the router-side
    /// pipeline's code for the same media: an empty url, a data URL whose
    /// header names no base64 payload, an empty payload (a URL part's or a
    /// base64 source's); a well-formed reference still forwards untouched, a
    /// data URL with parameters before the marker among them.
    #[test]
    fn a_reference_no_engine_serves_is_refused_at_the_door() {
        let components = components(MmProcessingMode::Worker);
        let registry = registry_with(vec![]);
        let worker_only = PlaceholderTokens::worker_only();
        let long = format!(" data:image/png;base64,{}", "A".repeat(200));
        let long_echo = format!("invalid media URL: {}...", &long[..64]);
        for (part, cause) in [
            (image_url(""), "invalid media URL: "),
            (image_url("   "), "invalid media URL:    "),
            (
                image_url(" data:image/png;base64,AQ=="),
                "invalid media URL:  data:image/png;base64,AQ==",
            ),
            (image_url(&long), long_echo.as_str()),
            (
                image_url("\u{1}data:image/png;base64,AQ=="),
                "invalid media URL: \u{1}data:image/png;base64,AQ==",
            ),
            (
                image_url("\u{7f}data:image/png;base64,AQ=="),
                "invalid media URL: \u{7f}data:image/png;base64,AQ==",
            ),
            (
                image_url("\u{9b}https://a/1.png"),
                "invalid media URL: \u{9b}https://a/1.png",
            ),
            (
                image_url("data:image/png,%89PNG%0D%0A"),
                "data URL parse error: only base64 encoded data URLs are supported",
            ),
            (
                image_url("data:image/png;base64;charset=x,AQ=="),
                "data URL parse error: only base64 encoded data URLs are supported",
            ),
            (
                image_url("data:image/png;base64,"),
                "data URL parse error: empty payload",
            ),
            (base64_image(""), "data URL parse error: empty payload"),
        ] {
            let error = resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![part]),
                &worker_only,
            )
            .expect_err(cause);
            assert_eq!(error.code(), "invalid_multimodal_request", "{cause}");
            assert_eq!(error.to_string(), format!("media item 0: {cause}"));
        }
        for part in [
            image_url("data:image/png;base64,AQ=="),
            image_url("data:image/png;charset=x;base64,AQ=="),
            image_url("data:image/png;charset=x;name=a.png;base64,AQ=="),
            image_url("https://a/1.png"),
            base64_image("AQ=="),
        ] {
            assert_eq!(
                resolve_mm_processing(
                    &components,
                    &registry,
                    MODEL,
                    &plan(vec![part]),
                    &worker_only
                )
                .expect("a well-formed reference forwards"),
                MmProcessing::Worker
            );
        }
    }

    #[test]
    fn schemes_label_is_parsed_case_insensitively() {
        let w = worker(
            "grpc://127.0.0.1:9200",
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[(MM_MEDIA_REF_SCHEMES_LABEL, "HTTP, https ,data,")],
        );
        let schemes = worker_media_ref_schemes(w.as_ref());
        assert_eq!(schemes.len(), 3);
        assert!(schemes.contains("http") && schemes.contains("https") && schemes.contains("data"));
    }

    /// A model without a registry spec: worker placement forwards it, auto
    /// forwards it when every worker of the model processes media itself and
    /// refuses it otherwise, router placement refuses it; the refusal names
    /// the router's missing spec, not the model.
    #[test]
    fn a_model_without_a_spec_goes_to_a_capable_worker_or_is_refused() {
        let worker_only = PlaceholderTokens::worker_only();
        let plan_ok = plan(vec![image_url("https://a/1.png")]);
        assert!(worker_only.worker_expandable(Modality::Image));

        assert_eq!(
            resolve_mm_processing(
                &components(MmProcessingMode::Worker),
                &registry_with(vec![]),
                MODEL,
                &plan_ok,
                &worker_only
            )
            .expect("worker placement forwards"),
            MmProcessing::Worker
        );
        let auto = components(MmProcessingMode::Auto);
        assert_eq!(
            resolve_mm_processing(
                &auto,
                &registry_with(vec![capable("grpc://127.0.0.1:9450")]),
                MODEL,
                &plan_ok,
                &worker_only
            )
            .expect("a uniform capable fleet forwards"),
            MmProcessing::Worker
        );
        for (registry, placement) in [
            (registry_with(vec![]), &auto),
            (
                registry_with(vec![capable("grpc://127.0.0.1:9451")]),
                &components(MmProcessingMode::Router),
            ),
        ] {
            let err = resolve_mm_processing(placement, &registry, MODEL, &plan_ok, &worker_only)
                .expect_err("nothing processes the media");
            assert_eq!(err, MmRefsError::RouterSpecMissing);
            assert_eq!(err.code(), "multimodal_not_supported");
            assert!(err.to_string().contains("no media spec"), "{err}");
        }
    }

    #[test]
    fn router_mode_never_forwards() {
        let registry = registry_with(vec![capable("grpc://127.0.0.1:9300")]);
        let resolved = resolve_mm_processing(
            &components(MmProcessingMode::Router),
            &registry,
            MODEL,
            &plan(vec![image_url("https://a/1.png")]),
            &expandable_placeholders(),
        )
        .expect("router mode resolves");
        assert_eq!(resolved, MmProcessing::Router);
    }

    #[test]
    fn auto_forwards_only_for_a_uniform_capable_fleet() {
        let placeholders = expandable_placeholders();
        let plan_ok = plan(vec![image_url("https://a/1.png")]);
        let components = components(MmProcessingMode::Auto);

        let uniform = registry_with(vec![
            capable("grpc://127.0.0.1:9400"),
            capable("grpc://127.0.0.1:9401"),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &uniform, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Worker
        );

        let mixed = registry_with(vec![
            capable("grpc://127.0.0.1:9402"),
            worker(
                "grpc://127.0.0.1:9403",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[],
            ),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &mixed, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Router
        );

        let empty = registry_with(vec![]);
        assert_eq!(
            resolve_mm_processing(&components, &empty, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Router
        );
    }

    /// An encode-prefill-decode pool resolves to router-side processing under
    /// `auto`, whatever its workers advertise: references never travel on
    /// that pipeline, so a pool whose workers all accept them would otherwise
    /// resolve to a mode selection refuses, with a 400 naming the mode the
    /// router already runs.
    #[test]
    fn auto_processes_in_the_router_for_an_encode_prefill_decode_pool() {
        let placeholders = expandable_placeholders();
        let plan_ok = plan(vec![image_url("https://a/1.png")]);
        let components = components(MmProcessingMode::Auto);
        let encode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:9410")
                .model(ModelCard::new(MODEL))
                .worker_type(WorkerType::Encode)
                .runtime_type(RuntimeType::Vllm)
                .connection_mode(ConnectionMode::Grpc)
                .label(MM_PROCESSOR_LABEL, "inprocess")
                .label(MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data")
                .health_config(no_health_check())
                .build(),
        );
        let epd = registry_with(vec![
            capable("grpc://127.0.0.1:9411"),
            capable("grpc://127.0.0.1:9412"),
            encode,
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &epd, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Router
        );
    }

    #[test]
    fn auto_falls_back_for_hints_and_unopted_models() {
        let components = components(MmProcessingMode::Auto);
        let registry = registry_with(vec![capable("grpc://127.0.0.1:9500")]);

        let hinted = plan(vec![MediaContentPart::ImageUrl {
            url: "https://a/1.png".to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: Some(512),
        }]);
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &hinted,
                &expandable_placeholders()
            )
            .expect("resolves"),
            MmProcessing::Router
        );

        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![image_url("https://a/1.png")]),
                &unopted
            )
            .expect("resolves"),
            MmProcessing::Router
        );
    }

    /// Under worker-side placement a model with workers and none that
    /// processes media references is refused in preparation, as the fleet
    /// verdict it is: a terminal 503 under the code selection has always
    /// answered the condition with, whose message names the flag and the
    /// fix. One capable worker lifts it (selection picks it); an empty pool
    /// is left to selection; a hinted request the router can serve keeps its
    /// router fallback; a fault of the request keeps its 400.
    #[test]
    fn worker_mode_refuses_a_fleet_that_processes_no_media_references() {
        use crate::routers::common::retry::is_retryable_response;

        let components = components(MmProcessingMode::Worker);
        let plan_ok = plan(vec![image_url("https://a/1.png")]);
        let placeholders = expandable_placeholders();
        let plain = |url: &str| worker(url, RuntimeType::Vllm, ConnectionMode::Grpc, &[]);

        let none_capable = registry_with(vec![
            plain("grpc://127.0.0.1:9800"),
            plain("grpc://127.0.0.1:9801"),
        ]);
        let err = resolve_mm_processing(&components, &none_capable, MODEL, &plan_ok, &placeholders)
            .expect_err("no worker processes media references");
        assert_eq!(
            err,
            MmRefsError::NoCapableWorker {
                model: MODEL.to_string(),
                workers: 2,
            }
        );
        assert_eq!(err.code(), "no_media_ref_capable_worker");
        let message = err.to_string();
        assert!(
            message.contains("--mm-processing worker") && message.contains("2 registered worker"),
            "{message}"
        );
        let response = err.into_response();
        assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers()[error::HEADER_X_SMG_ERROR_CODE],
            "no_media_ref_capable_worker"
        );
        assert!(
            !is_retryable_response(&response),
            "a capability verdict is terminal for the retry layer"
        );
        assert!(
            placement::refusal_reason(&response).is_some(),
            "the refusal's line names its reason"
        );

        // An engine of another family never processes references either.
        let other_engine = registry_with(vec![worker(
            "grpc://127.0.0.1:9802",
            RuntimeType::TokenSpeed,
            ConnectionMode::Grpc,
            &[],
        )]);
        assert!(matches!(
            resolve_mm_processing(&components, &other_engine, MODEL, &plan_ok, &placeholders),
            Err(MmRefsError::NoCapableWorker { workers: 1, .. })
        ));

        // A pool with an encode worker runs the encode-prefill-decode
        // pipeline, on which media references never travel whatever the
        // workers advertise: left to selection, whose refusal is the
        // request's own 400 and names the fix that applies there.
        let encode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:9805")
                .model(ModelCard::new(MODEL))
                .worker_type(WorkerType::Encode)
                .runtime_type(RuntimeType::Vllm)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        );
        let epd = registry_with(vec![plain("grpc://127.0.0.1:9806"), encode]);
        assert_eq!(
            resolve_mm_processing(&components, &epd, MODEL, &plan_ok, &placeholders)
                .expect("an encode-prefill-decode pool is selection's to refuse"),
            MmProcessing::Worker
        );

        let mixed = registry_with(vec![
            plain("grpc://127.0.0.1:9803"),
            capable("grpc://127.0.0.1:9804"),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &mixed, MODEL, &plan_ok, &placeholders)
                .expect("one capable worker is enough: selection picks it"),
            MmProcessing::Worker
        );

        let hinted = plan(vec![MediaContentPart::ImageUrl {
            url: "https://a/1.png".to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: Some(512),
        }]);
        assert_eq!(
            resolve_mm_processing(&components, &none_capable, MODEL, &hinted, &placeholders)
                .expect("the router applies the hint itself"),
            MmProcessing::Router
        );

        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        let fault = resolve_mm_processing(&components, &mixed, MODEL, &plan_ok, &unopted)
            .expect_err("an unopted model is the request's fault");
        assert_eq!(
            fault.code(),
            "multimodal_worker_processing_unsupported_model"
        );
        assert_eq!(
            fault.into_response().status(),
            http::StatusCode::BAD_REQUEST
        );
    }

    /// What readiness names: under worker-side placement, a model with a
    /// worker that reports a vision encoder and no worker that processes
    /// media references. A pool without the label (unknown) or with a
    /// `false` (a text-only engine) is not it, one capable worker takes the
    /// model off the list, a model with an encode worker is never on it (nor
    /// is its registration warned about), and the other placements name
    /// nothing.
    #[test]
    fn models_without_media_processing_names_vision_pools_with_no_capable_worker() {
        fn typed_worker_of(
            model: &str,
            url: &str,
            labels: &[(&str, &str)],
            worker_type: WorkerType,
        ) -> Arc<dyn Worker> {
            let mut builder = BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .worker_type(worker_type)
                .runtime_type(RuntimeType::Vllm)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check());
            for (key, value) in labels {
                builder = builder.label(*key, *value);
            }
            Arc::new(builder.build())
        }
        fn worker_of(model: &str, url: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
            typed_worker_of(model, url, labels, WorkerType::Regular)
        }
        let vision = [(SUPPORTS_VISION_LABEL, "true")];
        let vision_capable = [
            (SUPPORTS_VISION_LABEL, "true"),
            (MM_PROCESSOR_LABEL, "inprocess"),
        ];
        let text_only = [(SUPPORTS_VISION_LABEL, "false")];

        let workers: Vec<Arc<dyn Worker>> = vec![
            // Two vision workers, neither processes references: held.
            worker_of("vl-b", "grpc://127.0.0.1:9900", &vision),
            worker_of("vl-b", "grpc://127.0.0.1:9901", &vision),
            // A vision worker without, beside one with: the model is served.
            worker_of("vl-a", "grpc://127.0.0.1:9902", &vision),
            worker_of("vl-a", "grpc://127.0.0.1:9903", &vision_capable),
            // Text-only and unlabeled pools say nothing about media.
            worker_of("text", "grpc://127.0.0.1:9904", &text_only),
            worker_of("legacy", "grpc://127.0.0.1:9905", &[]),
            // An encode-prefill-decode pool: references never travel on it,
            // so its vision sides without a processor are not what readiness
            // waits on.
            typed_worker_of(
                "vl-epd",
                "grpc://127.0.0.1:9906",
                &vision,
                WorkerType::Prefill,
            ),
            typed_worker_of(
                "vl-epd",
                "grpc://127.0.0.1:9907",
                &vision,
                WorkerType::Decode,
            ),
            typed_worker_of("vl-epd", "grpc://127.0.0.1:9908", &[], WorkerType::Encode),
        ];
        assert_eq!(
            models_without_media_processing(MmProcessingMode::Worker, &workers),
            vec!["vl-b".to_string()]
        );
        // The registration warning follows the same line: vl-b's worker is
        // said, the encode-prefill-decode pool's vision side is not, and a
        // capable worker beside a plain one is not either; the plain one
        // beside it is said, with the pool serving the media (the line's
        // consequence reads the pool, `pool_processes_media_refs`).
        let pool_of = |model: &str| {
            workers
                .iter()
                .filter(move |worker| worker.model_id() == model)
                .collect::<Vec<_>>()
        };
        assert!(registration_lacks_media_processing(
            workers[0].as_ref(),
            pool_of("vl-b").into_iter()
        ));
        assert!(!registration_lacks_media_processing(
            workers[6].as_ref(),
            pool_of("vl-epd").into_iter()
        ));
        assert!(!registration_lacks_media_processing(
            workers[3].as_ref(),
            pool_of("vl-a").into_iter()
        ));
        assert!(registration_lacks_media_processing(
            workers[2].as_ref(),
            pool_of("vl-a").into_iter()
        ));
        assert!(pool_processes_media_refs(pool_of("vl-a").into_iter()));
        assert!(!pool_processes_media_refs(pool_of("vl-b").into_iter()));
        assert!(pool_has_encode_worker(pool_of("vl-epd").into_iter()));
        assert!(!pool_has_encode_worker(pool_of("vl-a").into_iter()));
        for placement in [MmProcessingMode::Auto, MmProcessingMode::Router] {
            assert!(models_without_media_processing(placement, &workers).is_empty());
        }
        assert!(models_without_media_processing(MmProcessingMode::Worker, &[]).is_empty());

        // The predicate behind it, per worker.
        assert!(vision_worker_without_media_processing(workers[0].as_ref()));
        assert!(!vision_worker_without_media_processing(workers[3].as_ref()));
        assert!(!vision_worker_without_media_processing(workers[4].as_ref()));
        assert!(!vision_worker_without_media_processing(workers[5].as_ref()));
    }

    #[test]
    fn worker_mode_is_strict() {
        let components = components(MmProcessingMode::Worker);
        // No registered workers: the decision is still Worker; selection enforces.
        let registry = registry_with(vec![]);
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![image_url("https://a/1.png")]),
                &expandable_placeholders()
            )
            .expect("resolves"),
            MmProcessing::Worker
        );

        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        let err = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![image_url("https://a/1.png")]),
            &unopted,
        )
        .expect_err("unopted model is refused");
        assert_eq!(err.code(), "multimodal_worker_processing_unsupported_model");

        let inline = plan(vec![MediaContentPart::ImageData {
            data: vec![0, 1, 2],
            mime_type: None,
            uuid: None,
            detail: None,
        }]);
        let err = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &inline,
            &expandable_placeholders(),
        )
        .expect_err("inline bytes are refused by name");
        assert_eq!(err, MmRefsError::UnsupportedPart("inline image bytes"));
    }

    fn hinted_image(url: &str, max_long_side_pixel: u32) -> MediaContentPart {
        MediaContentPart::ImageUrl {
            url: url.to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: Some(max_long_side_pixel),
        }
    }

    fn hinted_video(
        url: &str,
        fps: Option<f64>,
        max_long_side_pixel: Option<u32>,
    ) -> MediaContentPart {
        MediaContentPart::VideoUrl {
            url: url.to_string().into(),
            uuid: None,
            fps,
            max_long_side_pixel,
        }
    }

    #[test]
    fn worker_mode_preprocesses_hinted_items_itself() {
        let components = components(MmProcessingMode::Worker);
        let registry = registry_with(vec![capable("grpc://127.0.0.1:9700")]);
        let placeholders = expandable_placeholders();

        // The hints are the router pipeline's knobs: a request carrying one
        // is preprocessed here, whichever item carries it.
        for (name, hinted) in [
            (
                "image long side",
                plan(vec![hinted_image("https://a/1.png", 504)]),
            ),
            (
                "video fps",
                plan(vec![hinted_video("https://a/c.mp4", Some(1.0), None)]),
            ),
            (
                "video long side",
                plan(vec![hinted_video("https://a/c.mp4", None, Some(504))]),
            ),
            (
                "one hinted item among plain ones",
                plan(vec![
                    image_url("https://a/1.png"),
                    hinted_video("https://a/c.mp4", Some(2.0), None),
                ]),
            ),
        ] {
            assert_eq!(
                resolve_mm_processing(&components, &registry, MODEL, &hinted, &placeholders)
                    .unwrap_or_else(|err| panic!("{name}: {err}")),
                MmProcessing::Router,
                "{name}"
            );
        }

        // Plain items still forward.
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![image_url("https://a/1.png")]),
                &placeholders
            )
            .expect("plain items forward"),
            MmProcessing::Worker
        );

        // The router expands the placeholders itself, so the model's worker
        // expansion is not consulted for a hinted request.
        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![hinted_image("https://a/1.png", 252)]),
                &unopted
            )
            .expect("the router processes it"),
            MmProcessing::Router
        );

        // A part with no reference form is still refused by name, before or
        // after the hinted one.
        let inline = MediaContentPart::ImageData {
            data: vec![0, 1, 2],
            mime_type: None,
            uuid: None,
            detail: None,
        };
        for parts in [
            vec![hinted_image("https://a/1.png", 504), inline.clone()],
            vec![inline.clone(), hinted_image("https://a/1.png", 504)],
        ] {
            assert_eq!(
                resolve_mm_processing(&components, &registry, MODEL, &plan(parts), &placeholders)
                    .expect_err("inline bytes are refused"),
                MmRefsError::UnsupportedPart("inline image bytes")
            );
        }

        // Without a spec the router cannot apply the hints: the request is
        // refused for the hint, not for the missing spec.
        let err = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![hinted_image("https://a/1.png", 504)]),
            &PlaceholderTokens::worker_only(),
        )
        .expect_err("no pipeline applies the hint");
        assert_eq!(err.code(), "multimodal_hint_unsupported_in_worker_mode");
        assert!(
            err.to_string().contains("cannot apply them itself"),
            "{err}"
        );

        // vLLM also accepts the router's audio tensors. A hinted request
        // carrying audio is processed locally so its image hint takes effect;
        // the same audio without a hint still forwards.
        let audio = MediaContentPart::AudioUrl {
            url: "https://a/x.wav".to_string().into(),
            uuid: None,
        };
        let mut with_audio = expandable_placeholders();
        with_audio.insert(Modality::Audio, "<|audio_pad|>".to_string());
        with_audio.set_worker_expandable(Modality::Audio, true);
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![hinted_image("https://a/1.png", 504), audio.clone()]),
                &with_audio,
            )
            .expect("the router can apply the hint and serve audio"),
            MmProcessing::Router
        );
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![audio]),
                &with_audio
            )
            .expect("plain audio forwards"),
            MmProcessing::Worker
        );
    }

    /// The scheme refusal's remedy follows the refused scheme: a file URL
    /// names the worker's local-media flag, a data URL (an inline image the
    /// worker's labels leave out) gets the accepted list alone, and the
    /// sentence's head is the same for both.
    #[test]
    fn the_scheme_refusal_names_the_file_remedy_for_a_file_url_alone() {
        let file = MmRefsError::SchemeNotAccepted {
            scheme: "file".to_string(),
            accepted: "data,http,https".to_string(),
        }
        .to_string();
        assert_eq!(
            file,
            "the selected worker does not fetch 'file' URLs (accepts: data,http,https); \
             file:// needs --allowed-local-media-path on the worker"
        );
        let data = MmRefsError::SchemeNotAccepted {
            scheme: "data".to_string(),
            accepted: "http,https".to_string(),
        }
        .to_string();
        assert_eq!(
            data,
            "the selected worker does not fetch 'data' URLs (accepts: http,https)"
        );
        assert!(!data.contains("allowed-local-media-path"));
    }

    /// The empty scheme of a bare path or a port-less host written without a
    /// scheme (what the scheme read gives them; a host with a port reads as a
    /// scheme of its own) is refused like any scheme the worker's labels leave
    /// out: the accepted list alone, no local-media remedy, since a scheme-less
    /// string is not a file URL.
    #[test]
    fn the_scheme_refusal_gives_an_empty_scheme_the_accepted_list_alone() {
        assert_eq!(url_scheme("images/a.png"), "");
        assert_eq!(url_scheme("example.com/a.png"), "");
        assert_eq!(url_scheme("localhost:8000/a.png"), "localhost");
        let bare = MmRefsError::SchemeNotAccepted {
            scheme: String::new(),
            accepted: "http,https".to_string(),
        }
        .to_string();
        assert_eq!(
            bare,
            "the selected worker does not fetch '' URLs (accepts: http,https)"
        );
        assert!(!bare.contains("allowed-local-media-path"));
    }

    #[test]
    fn selection_check_requires_every_worker_and_scheme() {
        let plan_https = plan(vec![image_url("https://a/1.png")]);
        let single = WorkerSelection::Single {
            worker: capable("grpc://127.0.0.1:9600").into(),
        };
        ensure_selection_supports_media_refs(&single, &plan_https, PipelinePin::Any)
            .expect("capable single");

        let plain = WorkerSelection::Single {
            worker: worker(
                "grpc://127.0.0.1:9601",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[],
            )
            .into(),
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&plain, &plan_https, PipelinePin::Any),
            Err(MmRefsError::WorkerNotCapable)
        );

        let file_plan = plan(vec![image_url("file:///media/1.png")]);
        let err = ensure_selection_supports_media_refs(&single, &file_plan, PipelinePin::Any)
            .expect_err("file scheme not advertised");
        assert_eq!(err.code(), "media_ref_scheme_not_accepted");

        let pd = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: capable("grpc://127.0.0.1:9602"),
            decode: worker(
                "grpc://127.0.0.1:9603",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[],
            )
            .into(),
            runtime_type: RuntimeType::Vllm,
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&pd, &plan_https, PipelinePin::Any),
            Err(MmRefsError::WorkerNotCapable)
        );

        // Both workers fetch: the accepted schemes are the intersection of what
        // each advertising worker fetches.
        let file_prefill = worker(
            "grpc://127.0.0.1:9604",
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[
                (MM_PROCESSOR_LABEL, "inprocess"),
                (MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data,file"),
            ],
        );
        let mixed = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: file_prefill,
            decode: capable("grpc://127.0.0.1:9605").into(),
            runtime_type: RuntimeType::Vllm,
        };
        ensure_selection_supports_media_refs(&mixed, &plan_https, PipelinePin::Any)
            .expect("https on both workers");
        assert_eq!(
            ensure_selection_supports_media_refs(&mixed, &file_plan, PipelinePin::Any),
            Err(MmRefsError::SchemeNotAccepted {
                scheme: "file".to_string(),
                accepted: "data,http,https".to_string(),
            })
        );

        // A worker that advertises no schemes leaves the other worker's set in force.
        let silent_decode = worker(
            "grpc://127.0.0.1:9606",
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[(MM_PROCESSOR_LABEL, "inprocess")],
        );
        let half = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: capable("grpc://127.0.0.1:9607"),
            decode: silent_decode.into(),
            runtime_type: RuntimeType::Vllm,
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&half, &file_plan, PipelinePin::Any)
                .expect_err("prefill's set still applies")
                .code(),
            "media_ref_scheme_not_accepted"
        );
    }

    /// A worker running the gateway's own pipeline, which expands any
    /// spec's anchor itself.
    fn pipeline_worker(url: &str) -> Arc<dyn Worker> {
        worker(
            url,
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[
                (MM_PROCESSOR_LABEL, GATEWAY_PIPELINE_PROCESSOR),
                (MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data"),
            ],
        )
    }

    /// A spec'd model whose image anchor the engine's own processor does not
    /// expand.
    fn unopted_placeholders() -> PlaceholderTokens {
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, "<|media_pad|>".to_string());
        placeholders
    }

    /// Only a spec'd model with a modality the engine's processor does not
    /// expand needs the pipeline pin: an opted-in modality and a model
    /// without a spec (whose template renders the engine's own placeholders)
    /// go to any worker that processes references.
    #[test]
    fn only_unopted_modalities_of_a_specd_model_need_the_pipeline() {
        let image = plan(vec![image_url("https://a/1.png")]);
        assert!(plan_needs_gateway_pipeline(&image, &unopted_placeholders()));
        assert!(!plan_needs_gateway_pipeline(
            &image,
            &expandable_placeholders()
        ));
        assert!(!plan_needs_gateway_pipeline(
            &image,
            &PlaceholderTokens::worker_only()
        ));
        assert!(worker_runs_gateway_pipeline(
            pipeline_worker("grpc://127.0.0.1:9700").as_ref()
        ));
        assert!(!worker_runs_gateway_pipeline(
            capable("grpc://127.0.0.1:9701").as_ref()
        ));

        // The pin selection reads: the pipeline alone for an unopted anchor,
        // never the pipeline for audio, either otherwise.
        let audio = plan(vec![MediaContentPart::AudioUrl {
            url: "https://a/x.wav".to_string().into(),
            uuid: None,
        }]);
        let mut audio_opted = PlaceholderTokens::default();
        audio_opted.insert(Modality::Audio, "<|audio_pad|>".to_string());
        audio_opted.set_worker_expandable(Modality::Audio, true);
        assert_eq!(
            media_refs_pipeline_pin(&image, &unopted_placeholders()),
            PipelinePin::Only
        );
        assert_eq!(
            media_refs_pipeline_pin(&audio, &audio_opted),
            PipelinePin::Never
        );
        assert_eq!(
            media_refs_pipeline_pin(&image, &expandable_placeholders()),
            PipelinePin::Any
        );
        assert_eq!(
            media_refs_pipeline_pin(&image, &PlaceholderTokens::worker_only()),
            PipelinePin::Any
        );
    }

    /// Strict worker mode forwards a model the engine's processor cannot
    /// expand when every media-processing worker of the model runs the
    /// gateway's own pipeline; a fleet mixing in the engine's processors, or
    /// one with no such worker yet, keeps the refusal that names the opt-in.
    #[test]
    fn worker_mode_forwards_unopted_models_to_a_pipeline_fleet() {
        let components = components(MmProcessingMode::Worker);
        let image = plan(vec![image_url("https://a/1.png")]);
        let unopted = unopted_placeholders();

        let pipeline_fleet = registry_with(vec![
            pipeline_worker("grpc://127.0.0.1:9710"),
            pipeline_worker("grpc://127.0.0.1:9711"),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &pipeline_fleet, MODEL, &image, &unopted)
                .expect("forwarded to the pipeline workers"),
            MmProcessing::Worker
        );

        let mixed = registry_with(vec![
            pipeline_worker("grpc://127.0.0.1:9712"),
            capable("grpc://127.0.0.1:9713"),
        ]);
        let err = resolve_mm_processing(&components, &mixed, MODEL, &image, &unopted)
            .expect_err("a worker on the engine's processor keeps the opt-in");
        assert_eq!(err, MmRefsError::ModelNotOptedIn(Modality::Image));
        assert!(
            err.to_string().contains(GATEWAY_PIPELINE_PROCESSOR),
            "{err}"
        );

        let empty = registry_with(vec![]);
        assert_eq!(
            resolve_mm_processing(&components, &empty, MODEL, &image, &unopted)
                .expect_err("no fleet"),
            MmRefsError::ModelNotOptedIn(Modality::Image)
        );

        // The pipeline takes no audio, so an unopted audio anchor is refused
        // here, with the opt-in named, rather than forwarded to a refusal.
        let audio = plan(vec![MediaContentPart::AudioUrl {
            url: "https://a/x.wav".to_string().into(),
            uuid: None,
        }]);
        let mut unopted_audio = PlaceholderTokens::default();
        unopted_audio.insert(Modality::Audio, "<|audio_pad|>".to_string());
        assert_eq!(
            resolve_mm_processing(&components, &pipeline_fleet, MODEL, &audio, &unopted_audio)
                .expect_err("the pipeline expands no audio anchor"),
            MmRefsError::ModelNotOptedIn(Modality::Audio)
        );

        // An opted-in model needs no pipeline worker, as before.
        assert_eq!(
            resolve_mm_processing(
                &components,
                &mixed,
                MODEL,
                &image,
                &expandable_placeholders()
            )
            .expect("opted-in model forwards to any capable worker"),
            MmProcessing::Worker
        );
    }

    /// `auto` forwards an unopted model only to a fleet made entirely of
    /// pipeline workers; one worker on the engine's processor sends the
    /// model back to the gateway's own pipeline, as before.
    #[test]
    fn auto_forwards_unopted_models_only_to_a_uniform_pipeline_fleet() {
        let components = components(MmProcessingMode::Auto);
        let image = plan(vec![image_url("https://a/1.png")]);
        let unopted = unopted_placeholders();

        let pipeline_fleet = registry_with(vec![
            pipeline_worker("grpc://127.0.0.1:9720"),
            pipeline_worker("grpc://127.0.0.1:9721"),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &pipeline_fleet, MODEL, &image, &unopted)
                .expect("resolves"),
            MmProcessing::Worker
        );

        let mixed = registry_with(vec![
            pipeline_worker("grpc://127.0.0.1:9722"),
            capable("grpc://127.0.0.1:9723"),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &mixed, MODEL, &image, &unopted).expect("resolves"),
            MmProcessing::Router
        );
    }

    /// Selection pins an unopted model's references to pipeline workers on
    /// every worker, and keeps audio off a pipeline worker, which takes none.
    #[test]
    fn selection_check_pins_unopted_models_to_pipeline_workers_and_audio_off_them() {
        let image = plan(vec![image_url("https://a/1.png")]);
        let audio = plan(vec![MediaContentPart::AudioUrl {
            url: "https://a/x.wav".to_string().into(),
            uuid: None,
        }]);
        let pipeline_single = WorkerSelection::Single {
            worker: pipeline_worker("grpc://127.0.0.1:9730").into(),
        };
        let engine_single = WorkerSelection::Single {
            worker: capable("grpc://127.0.0.1:9731").into(),
        };
        ensure_selection_supports_media_refs(&pipeline_single, &image, PipelinePin::Only)
            .expect("the pipeline worker takes the unopted model");
        assert_eq!(
            ensure_selection_supports_media_refs(&engine_single, &image, PipelinePin::Only),
            Err(MmRefsError::PipelineWorkerRequired)
        );
        ensure_selection_supports_media_refs(&engine_single, &image, PipelinePin::Any)
            .expect("an opted-in model goes to the engine's processor");

        let half = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: pipeline_worker("grpc://127.0.0.1:9732"),
            decode: capable("grpc://127.0.0.1:9733").into(),
            runtime_type: RuntimeType::Vllm,
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&half, &image, PipelinePin::Only),
            Err(MmRefsError::PipelineWorkerRequired)
        );

        assert_eq!(
            ensure_selection_supports_media_refs(&pipeline_single, &audio, PipelinePin::Any),
            Err(MmRefsError::PipelineTakesNoAudio)
        );
        ensure_selection_supports_media_refs(&engine_single, &audio, PipelinePin::Any)
            .expect("the engine's processor takes audio");
        assert_eq!(
            MmRefsError::PipelineTakesNoAudio.code(),
            "multimodal_not_supported"
        );
        assert_eq!(
            MmRefsError::PipelineWorkerRequired.code(),
            "multimodal_worker_processing_unsupported_model"
        );
    }

    #[test]
    fn assemble_keeps_prompt_order_and_maps_modalities() {
        let refs = assemble_media_refs(&plan(vec![
            image_url("https://a/1.png"),
            MediaContentPart::VideoUrl {
                url: "data:video/mp4;base64,AAAA".to_string().into(),
                uuid: None,
                fps: None,
                max_long_side_pixel: None,
            },
            image_url("https://a/2.png"),
        ]))
        .expect("assembles");
        let items: Vec<(i32, &str)> = refs
            .items
            .iter()
            .map(|item| (item.modality, item.url.as_str()))
            .collect();
        assert_eq!(
            items,
            vec![
                (common::Modality::Image as i32, "https://a/1.png"),
                (common::Modality::Video as i32, "data:video/mp4;base64,AAAA"),
                (common::Modality::Image as i32, "https://a/2.png"),
            ]
        );
    }

    #[test]
    fn assemble_builds_the_data_url_of_a_base64_part() {
        let refs = assemble_media_refs(&plan(vec![base64_image("AAAA"), base64_audio("BBBB")]))
            .expect("assembles");
        let urls: Vec<&str> = refs.items.iter().map(|item| item.url.as_str()).collect();
        assert_eq!(
            urls,
            vec!["data:image/png;base64,AAAA", "data:audio/wav;base64,BBBB"]
        );
    }

    /// Hint detection classifies the parts and builds no URL: the data URL of
    /// a base64 part is built once, at assembly, so detection allocates nothing
    /// however large the payloads are.
    #[cfg(all(
        feature = "jemalloc-stats",
        not(target_env = "msvc"),
        not(target_env = "musl")
    ))]
    #[test]
    fn hint_detection_builds_no_data_url() {
        let plan = plan(vec![
            base64_image(&"A".repeat(4 << 20)),
            base64_audio(&"B".repeat(1 << 20)),
        ]);
        // Warm the one-time caches on the path (the env-read byte caps) so
        // the counter measures the per-request cost alone.
        assert_eq!(plan_carries_hints(&plan), Ok(false));
        let bytes = bytes_allocated_during(|| {
            assert_eq!(plan_carries_hints(&plan), Ok(false));
        });
        assert_eq!(bytes, 0, "detection allocated {bytes} bytes");
    }

    #[test]
    fn audio_refs_require_an_expandable_model_and_stay_in_prompt_order() {
        let plan = plan(vec![
            MediaContentPart::AudioUrl {
                url: "data:audio/wav;base64,AAAA".to_string().into(),
                uuid: None,
            },
            image_url("https://a/1.png"),
        ]);
        assert!(plan.is_forwardable());
        let mut placeholders = expandable_placeholders();
        assert_eq!(
            ensure_worker_expandable(&plan, &placeholders, false),
            Err(MmRefsError::ModelNotOptedIn(Modality::Audio))
        );
        placeholders.set_worker_expandable(Modality::Audio, true);
        ensure_worker_expandable(&plan, &placeholders, false).unwrap();
        let refs = assemble_media_refs(&plan).unwrap();
        assert_eq!(refs.items[0].modality, common::Modality::Audio as i32);
        assert_eq!(refs.items[0].url, "data:audio/wav;base64,AAAA");
        assert_eq!(refs.items[1].modality, common::Modality::Image as i32);
    }

    #[test]
    fn assemble_rejects_unsupported_parts_and_oversized_data_urls() {
        let err = assemble_media_refs(&plan(vec![MediaContentPart::AudioData {
            data: vec![0; 4],
            mime_type: Some("audio/wav".to_string()),
            uuid: None,
        }]))
        .expect_err("raw audio bytes have no reference form");
        assert_eq!(err, MmRefsError::UnsupportedPart("inline audio bytes"));

        let oversized = format!(
            "data:image/png;base64,{}",
            "A".repeat(media::image_max_input_bytes() * 4 / 3 + 8)
        );
        let err = assemble_media_refs(&plan(vec![image_url(&oversized)]))
            .expect_err("oversized data url");
        assert_eq!(err.code(), "media_ref_too_large");
    }

    #[test]
    fn url_scheme_and_data_url_size() {
        assert_eq!(url_scheme("HTTPS://a/1.png"), "https");
        assert_eq!(url_scheme("data:image/png;base64,AAAA"), "data");
        assert_eq!(url_scheme("/local/path.png"), "");
        assert_eq!(data_url_payload_bytes("https://a/1.png"), None);
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAAAAAAA"),
            Some(6)
        );
        assert_eq!(data_url_payload_bytes("DATA:text/plain,hello"), Some(5));
    }
}
