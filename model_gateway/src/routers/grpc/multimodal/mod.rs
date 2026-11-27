//! Multimodal processing integration for gRPC pipeline (chat + messages).
//!
//! Bridges the `llm-multimodal` crate with the gRPC router pipeline, split by
//! processing phase:
//!
//! - [`detect`]: find modalities and extract content parts from chat/messages.
//! - [`config`]: model config-file registry and per-router component bundle.
//! - [`resolve`]: what a request's media needs before the template renders:
//!   the placeholder tokens for its plan and the model's media-part order.
//! - [`process`]: fetch media → preprocess → expand placeholder tokens →
//!   build the lightweight [`MultimodalIntermediate`].
//! - [`assemble`]: turn the intermediate into backend-specific `MultimodalData`
//!   once the target backend is known (after worker selection).
//! - [`serialize`]: tensor byte/dtype serialization used by assembly.
//! - [`transport`]: SHM-vs-inline transport resolution and `/dev/shm`
//!   namespace verification.
//! - [`refs`]: router-vs-worker processing resolution and the media-reference
//!   payload for workers that process media themselves.

use std::{collections::HashSet, fmt::Display, sync::Arc};

use axum::response::Response;
use llm_multimodal::{
    hub::HubFetchError, AudioClip, EncoderFieldLayouts, EncoderInputFormat, ImageFrame,
    MediaConnectorError, MediaErrorCode, MediaFault, Modality, MultiModalError, PlaceholderRange,
    PreprocessedEncoderInputs, VideoClip,
};
use llm_tokenizer::{Encoding, TokenizerTrait};

use crate::routers::{
    common::retry::mark_non_retryable, error, grpc::proto_wrapper::VLLM_MROPE_GRID_KEYS,
};

/// Bridges the gateway's tokenizer to the model registry's local tokenizer
/// seam: `llm-multimodal` deliberately depends on no tokenizer crate, so the
/// flattening from [`Encoding`] to plain ids lives here, on the caller side.
pub(crate) struct RegistryTokenizer<'a>(pub(crate) &'a dyn TokenizerTrait);

impl llm_multimodal::Tokenizer for RegistryTokenizer<'_> {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }

    fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
        Some(match self.0.encode(text, false).ok()? {
            Encoding::Hf(inner) => inner.get_ids().to_vec(),
            Encoding::Plain(ids) | Encoding::Tiktoken(ids) => ids,
        })
    }
}

mod assemble;
mod capability;
mod config;
mod detect;
pub(crate) mod ec_push;
mod encoder_budget;
mod inflight;
mod item_limits;
mod pixel_cache;
mod plan;
mod process;
mod refs;
mod resolve;
mod serialize;
mod settings;
mod transparency;
mod transport;
pub mod worker;

#[cfg(test)]
pub(crate) use assemble::tests::single_image_intermediate;
pub(crate) use assemble::{
    assemble_multimodal_data, assemble_multimodal_data_after_encode,
    assemble_tokenspeed_for_encode, encode_routing_hashes,
};
pub(crate) use capability::{
    ensure_backend_supports_modalities, ensure_model_accepts_media, fleet_reads_format,
    request_needs_vision, worker_language_model_only, MediaFacts,
};
#[cfg(test)]
pub(crate) use config::tests::{hub_answering, HubEnv};
pub(crate) use config::{
    load_image_preprocessor_config, load_video_preprocessor_config, MultimodalComponents,
    MultimodalConfigRegistry, MultimodalModelConfig,
};
pub(crate) use detect::{media_plan_chat, media_plan_messages};
pub(crate) use encoder_budget::engine_encoder_budget;
pub(crate) use inflight::{
    reserve_multimodal_inflight, InflightPermit, UNIT_BYTES as INFLIGHT_UNIT_BYTES,
};
pub(crate) use item_limits::engine_item_limits;
pub(crate) use plan::{
    anchor_rendered_markers, prepare_placeholder_tokens, resolve_media_part_order,
    validate_rendered_media_anchors, DeploymentFault, MediaPlan, PlaceholderTokens, RequestRefusal,
};
pub(crate) use process::process_multimodal_plan;
#[cfg(test)]
pub(crate) use refs::MM_PROCESSOR_LABEL;
pub(crate) use refs::{
    assemble_media_refs, ensure_selection_supports_media_refs, media_refs_pipeline_pin,
    models_without_media_processing, pool_processes_media_refs,
    registration_lacks_media_processing, resolve_mm_processing, worker_accepts_media_refs,
    worker_runs_gateway_pipeline, worker_supports_ec_transfer, MmProcessing, MmRefsError,
    GATEWAY_PIPELINE_PROCESSOR,
};
pub(crate) use resolve::{resolve_media, MediaContext, ResolvedMedia};
pub(crate) use serialize::WireMismatch;
pub(crate) use settings::{init_mm_settings, mm_settings, MultimodalSettings};
pub(crate) use transparency::engine_transparency_policy;
pub(crate) use transport::{
    init_mm_transport_defaults, mm_image_wire_cap_over, mm_rdma_exporter,
    mm_shared_image_wire_dtype_over,
};

/// The response for an error of assembling or planning a request's
/// multimodal data: the typed wire mismatch is the gateway's own transient
/// state (the model's workers changed between preparation and selection),
/// answered with a retryable status; anything else is the request's.
pub(crate) fn assembly_error_response(error: &anyhow::Error) -> Response {
    if WireMismatch::is(error) {
        error::service_unavailable("multimodal_wire_changed", format!("{error}"))
    } else {
        error::bad_request("multimodal_not_supported", format!("{error}"))
    }
}

/// The client-facing message for a media pipeline failure: the route's
/// sentence and the pipeline's verdict on the request (the connector's own
/// words for the URL whose fetch ran out of its budget, the clip the decoder
/// refused, the cap a payload met), without the gateway's stage frames between
/// them: "failed to finalize the tracker" names a structure the client never
/// sees. The stage frames sit above the tracker's own typed error, so a
/// failure with no connector verdict reads that error's own text when the
/// chain carries one, and the whole chain otherwise: the frames that say what
/// failed (the config read, the replacement index, the panic) stay in the
/// client's sentence. The whole chain, stage frames included, is the log
/// line's at the call site.
pub(crate) fn processing_failed_message(error: &anyhow::Error) -> String {
    let cause = if let Some(media) = media_verdict(error) {
        media.to_string()
    } else if let Some(tracker) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<MultiModalError>())
    {
        tracker.to_string()
    } else {
        format!("{error:#}")
    };
    format!("Multimodal processing failed: {cause}")
}

/// The media connector's own error in a pipeline failure's chain, if any:
/// the verdict both pipelines answer the client by, the router-side one
/// directly ([`processing_failed_response`]) and the worker-side one through
/// the trailers of the worker's refusal (`worker::classify`).
pub(crate) fn media_verdict(error: &anyhow::Error) -> Option<&MediaConnectorError> {
    error.chain().find_map(|cause| {
        if let Some(media) = cause.downcast_ref::<MediaConnectorError>() {
            return Some(media);
        }
        match cause.downcast_ref::<MultiModalError>() {
            Some(MultiModalError::Media(media)) => Some(media),
            _ => None,
        }
    })
}

/// The client's answer for a failure of the router-side media pipeline, by
/// the connector's verdict on it, the same the worker-side pipeline reports
/// for the same failure: a failure of the request's own media (a reference it
/// got wrong, an origin that refuses or answers 4xx, a clip over a size cap)
/// is a 400 under the media's code (`invalid_multimodal_request` for media
/// refused as the request sent it, `media_fetch_failed`, `media_too_large`);
/// a failure of the media's host for now (an origin that does not deliver
/// within the fetch budget, answers 5xx or breaks the transfer) a 502 under
/// the same code, so a client can tell what to fix from what to retry, and
/// terminal for the retry layer (another attempt would meet the same host);
/// this process's own fault (a scratch file, a decoder it cannot run, a
/// decode task that died) a 500; a failure with no verdict in its chain (a
/// part the pipeline does not take, a model it has no spec for) stays a 400
/// `multimodal_processing_failed`.
/// An item the engine could never place ([`encoder_budget::ItemOverEncoderBudget`])
/// is a 400 under a code of its own, so a client can tell the item's size from a
/// fetch or a decode that failed. A video part on a process without a video
/// decoder ([`process::VideoDecoderUnavailable`], known at start-up) is the
/// media stage's own refusal, `400 invalid_multimodal_request` in the words
/// of a modality the model does not take. An image size the processor's
/// rule refuses ([`process::ImageSizeRefused`]) is a 400 under the same code,
/// in the preprocessing's words. A configuration file the model's
/// hub failed to serve is answered by [`config_unavailable_response`].
pub(crate) fn processing_failed_response(error: &anyhow::Error) -> Response {
    let message = processing_failed_message(error);
    if encoder_budget::ItemOverEncoderBudget::is(error) {
        return error::bad_request(encoder_budget::ITEM_OVER_ENCODER_BUDGET_CODE, message);
    }
    if process::VideoDecoderUnavailable::is(error) {
        return invalid_multimodal_request(format!("{error:#}"));
    }
    if process::ImageSizeRefused::is(error) {
        return error::bad_request(MediaErrorCode::InvalidRequest.as_str(), message);
    }
    if let Some(response) = config_unavailable_response(error) {
        return response;
    }
    if let Some(media) = media_verdict(error) {
        return match media.fault() {
            MediaFault::Client => error::bad_request(media.code().as_str(), message),
            MediaFault::Transient => {
                let mut response = error::bad_gateway(media.code().as_str(), message);
                mark_non_retryable(&mut response);
                response
            }
            MediaFault::Internal => error::internal_error(PIPELINE_CODE, message),
        };
    }
    let ours = error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<MultiModalError>(),
            Some(MultiModalError::Join(_))
        ) || cause.downcast_ref::<tokio::task::JoinError>().is_some()
    });
    if ours {
        error::internal_error(PIPELINE_CODE, message)
    } else {
        error::bad_request(PIPELINE_CODE, message)
    }
}

/// Whether `error` carries an explicit verdict that the request's own media
/// is at fault: the connector's client verdict, an item over the encoder's
/// budget, a video part on a process that cannot decode video, or an image
/// size the processor's rule refuses. A failure with no verdict in its chain
/// does not, though [`processing_failed_response`]
/// answers it 400 too: that class also holds a model configuration this host
/// could not load and a processor's own fault, which the client cannot fix.
pub(crate) fn is_request_media_fault(error: &anyhow::Error) -> bool {
    encoder_budget::ItemOverEncoderBudget::is(error)
        || process::VideoDecoderUnavailable::is(error)
        || process::ImageSizeRefused::is(error)
        || (!HubFetchError::is(error)
            && media_verdict(error).is_some_and(|media| media.fault() == MediaFault::Client))
}

/// The client's answer when the model's hub failed to serve a configuration
/// file ([`HubFetchError`] in the chain), on every path a request's media
/// is resolved: a 502 under `multimodal_config_unavailable` with a generic
/// message, since the fault is upstream of the gateway and the message of the
/// chain names the hub's address. The response is left retryable: nothing of
/// the model is kept from the failed attempt, so the next request resolves
/// the checkpoint again (unlike a media host's transient fault, which the
/// same host would repeat). The cause (the file, the hub's answer) is for the
/// log at the call site. `None` for any other failure.
pub(crate) fn config_unavailable_response(error: &anyhow::Error) -> Option<Response> {
    HubFetchError::is(error).then(|| {
        error::bad_gateway(
            "multimodal_config_unavailable",
            "The model's configuration could not be fetched from the model hub; \
             retry the request",
        )
    })
}

/// The code of a failure of the media pipeline that has no code of its own.
const PIPELINE_CODE: &str = "multimodal_processing_failed";

/// The client's answer when the placeholder preparation fails for a cause
/// its site marks as the deployment's own ([`DeploymentFault`]): a
/// `config.json` this gateway holds and cannot read or parse, a spec's
/// media limits or its placeholder token that cannot be resolved from the
/// model's files, a vocabulary that lacks the token. A 500 under the
/// pipeline's own code with a generic message: a client cannot fix it by
/// its request, and the cause (the file, the token) is for the log at the
/// call site. Unmarked failures from resolving the configuration directory
/// keep their existing answer; these include an absent or inaccessible
/// configuration as well as local client or cache setup failures.
pub(crate) fn preparation_fault_response() -> Response {
    error::internal_error(
        PIPELINE_CODE,
        "The gateway could not prepare media processing for this model; \
         the request is not at fault",
    )
}

/// The 400 for media the request carries that cannot be prepared, under the
/// media stage's code (`MediaErrorCode::InvalidRequest`, the one the media
/// pipeline answers media refused as the request sent it under).
pub(crate) fn invalid_multimodal_request(error: impl Display) -> Response {
    error::bad_request(
        MediaErrorCode::InvalidRequest.as_str(),
        format!("Invalid multimodal request: {error}"),
    )
}

/// Whether verbose multimodal timing logs are enabled (`--log-mm-timing` /
/// `SMG_LOG_MM_TIMING`). Resolved once at startup; called on every multimodal
/// request.
pub(crate) fn log_mm_timing_enabled() -> bool {
    mm_settings().log_mm_timing.value
}

/// Output of the multimodal processing pipeline.
pub(crate) struct MultimodalOutput {
    /// Token IDs with placeholder tokens expanded to the correct count per media item.
    pub expanded_token_ids: Vec<u32>,
    /// Lightweight intermediate holding preprocessing results.
    /// Assembled into backend-specific `MultimodalData` in request_building.
    pub intermediate: MultimodalIntermediate,
}

/// Lightweight intermediate from the preparation stage.
///
/// Holds all preprocessing results without serializing tensors to bytes.
/// The assembly stage converts this into a backend-specific `MultimodalData`
/// variant once the target backend is known (after worker selection).
#[derive(Debug)]
pub(crate) struct MultimodalIntermediate {
    /// Independently preprocessed modality batches sharing one expanded prompt.
    /// A single-modality request is represented by a one-element vector.
    batches: Vec<PrecomputedMultimodalIntermediate>,
    /// The format the spec's processors emitted the batches in
    /// (`ModelProcessorSpec::encoder_input_format`).
    encoder_input_format: EncoderInputFormat,
    /// The tokens the expansion wrote into the prompt for the request's
    /// images, summed over them (each image's whole replacement; the
    /// template's own markers around an anchor are text tokens): what the
    /// chat usage reports as `prompt_tokens_details.image_tokens`. `None`
    /// when no batch is images; a video's or an audio clip's placeholders
    /// are not image tokens.
    image_prompt_tokens: Option<u32>,
}

impl MultimodalIntermediate {
    pub(crate) fn try_new(batches: Vec<PrecomputedMultimodalIntermediate>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !batches.is_empty(),
            "multimodal intermediate requires at least one batch"
        );
        let mut modalities = HashSet::with_capacity(batches.len());
        for batch in &batches {
            let modality = batch.media.modality();
            anyhow::ensure!(
                modalities.insert(modality),
                "multimodal intermediate contains duplicate {modality} batches"
            );
            anyhow::ensure!(
                batch.media.len() > 0,
                "multimodal intermediate contains an empty {modality} batch"
            );
        }
        Ok(Self {
            batches,
            encoder_input_format: EncoderInputFormat::Reference,
            image_prompt_tokens: None,
        })
    }

    pub(crate) fn with_encoder_input_format(mut self, format: EncoderInputFormat) -> Self {
        self.encoder_input_format = format;
        self
    }

    pub(crate) fn encoder_input_format(&self) -> EncoderInputFormat {
        self.encoder_input_format
    }

    pub(crate) fn with_image_prompt_tokens(mut self, image_prompt_tokens: Option<u32>) -> Self {
        self.image_prompt_tokens = image_prompt_tokens;
        self
    }

    pub(crate) fn image_prompt_tokens(&self) -> Option<u32> {
        self.image_prompt_tokens
    }

    pub(crate) fn batches(&self) -> &[PrecomputedMultimodalIntermediate] {
        &self.batches
    }

    /// Whether a batch carries the grid tensors a disaggregated decode side
    /// derives its positions from ([`VLLM_MROPE_GRID_KEYS`]): such a side
    /// cannot be served by a decode worker without a vision encoder.
    pub(crate) fn has_mrope_grids(&self) -> bool {
        self.batches.iter().any(|batch| {
            batch
                .preprocessed
                .model_specific
                .keys()
                .any(|key| VLLM_MROPE_GRID_KEYS.contains(&key.as_str()))
        })
    }

    pub(crate) fn into_batches(self) -> Vec<PrecomputedMultimodalIntermediate> {
        self.batches
    }
}

/// Raw media for one preprocessed batch.
///
/// Encoding the modality in the enum prevents contradictory states such as an
/// audio batch carrying images or an image batch carrying both images and
/// videos.
#[derive(Debug, Clone)]
pub(crate) enum MediaBatch {
    Images(Vec<Arc<ImageFrame>>),
    Audios(Vec<Arc<AudioClip>>),
    Videos(Vec<Arc<VideoClip>>),
}

impl MediaBatch {
    pub(crate) fn modality(&self) -> Modality {
        match self {
            Self::Images(_) => Modality::Image,
            Self::Audios(_) => Modality::Audio,
            Self::Videos(_) => Modality::Video,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Images(items) => items.len(),
            Self::Audios(items) => items.len(),
            Self::Videos(items) => items.len(),
        }
    }
}

/// Explicit association between one media item and its expanded prompt span.
#[derive(Debug, Clone)]
pub(crate) struct PromptBinding {
    /// Index of the media/preprocessed item within its modality batch.
    pub item_index: usize,
    /// Position of this media item among all modalities in the rendered prompt.
    pub prompt_ordinal: usize,
    /// Full replacement span, including structural tokens.
    pub structural: PlaceholderRange,
    /// Patch-only spans within `structural`.
    pub patches: Vec<PlaceholderRange>,
}

#[derive(Debug)]
pub(crate) struct PrecomputedMultimodalIntermediate {
    /// Preprocessed encoder input and model-specific tensors (not yet serialized).
    pub preprocessed: PreprocessedEncoderInputs,
    /// Raw media whose variant determines this batch's modality.
    pub media: MediaBatch,
    /// Exact media-to-prompt associations for this batch.
    pub bindings: Vec<PromptBinding>,
    /// Placeholder token ID from model config for the active modality.
    pub placeholder_token_id: Option<u32>,
    /// Primary encoder input and model-specific side-tensor layouts.
    pub field_layouts: EncoderFieldLayouts,
    /// Tensor keys that should remain on CPU (vLLM `keep_on_cpu` hint).
    pub keep_on_cpu_keys: Vec<String>,
    /// Wire key for the primary encoder tensor when the model's forward does
    /// not take `pixel_values` (DeepSeek-V4.1 takes `patches`); `None` keeps
    /// the default name.
    pub encoder_input_key: Option<String>,
}

/// Sets the decoder availability gauges (`smg_mm_turbojpeg_available`,
/// `smg_mm_webp_available`) on the recorder in force and returns what they
/// read: the call [`report_jpeg_decoder`] makes every time it runs, kept
/// apart so a test can run it against a recorder of its own. The gateway
/// installs its Prometheus recorder before its report (`server.rs`): a gauge
/// set before that lands in the no-op recorder and never reaches /metrics,
/// which is why the gauges are set on every report and not once.
pub(crate) fn record_decoder_availability() -> (bool, bool) {
    use crate::observability::metrics::Metrics;
    let available = llm_multimodal::jpeg_turbo::turbojpeg_available();
    Metrics::set_mm_turbojpeg_available(available);
    let webp = llm_multimodal::webp_native::webp_available();
    Metrics::set_mm_webp_available(webp);
    (available, webp)
}

/// Say once which JPEG and WebP decoders this process has, and publish them:
/// without libjpeg-turbo the pure-Rust decoder's pixels differ from PIL's by a
/// few levels, an embedding shift against an engine that decodes with PIL;
/// without libwebp the WebP fallback is correct but several times slower.
pub(crate) fn report_jpeg_decoder() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    // The gauges on every call, the log lines once: a caller that builds the
    // multimodal components before the start-up installs its recorder would
    // otherwise consume the once block on the no-op recorder, and the
    // start-up's own call could set neither gauge.
    let (available, _webp) = record_decoder_availability();
    ONCE.get_or_init(|| {
        if available {
            tracing::info!("JPEG decode uses libjpeg-turbo with PIL's defaults");
        } else {
            tracing::warn!(
                "libturbojpeg not found: JPEGs decode with the pure-Rust decoder, whose pixels \
                 differ from PIL's by a few levels; install libjpeg-turbo for exact parity with \
                 an engine that decodes with PIL, or start with `--mm-turbojpeg require` to \
                 refuse to serve without it"
            );
        }
        match llm_multimodal::webp_native::webp_version() {
            Some(version) => tracing::info!(%version, "WebP decode uses libwebp"),
            None => tracing::info!(
                "libwebp not found: WebP files decode with the pure-Rust decoder (the same \
                 pixels, several times the cost); install libwebp for the faster decode"
            ),
        }
        match llm_multimodal::missing_video_decoder() {
            None => tracing::info!("video decode has its decoder"),
            Some(why) => tracing::warn!(
                "video input cannot be decoded by this process: {why}; a request with a video \
                 part is refused up front until it can"
            ),
        }
    });
}

/// Apply the operator's TurboJPEG policy to this process. Under `require` a
/// process whose loader did not find the library does not come up, with the
/// library and the gauge named, instead of serving the fallback decoder's
/// pixels behind the one warning above; under `prefer` nothing changes. The
/// router calls it once at start-up, in every routing mode, before anything
/// binds (a gRPC router created behind the gateway's try-register would
/// otherwise log its refusal and the process would start without it), and
/// again where its components and the worker pipeline are built, for an
/// embedded or test caller; the line under `require` is said once.
pub(crate) fn enforce_jpeg_decoder_policy(
    policy: llm_multimodal::jpeg_turbo::TurboJpegPolicy,
) -> anyhow::Result<()> {
    static SAID: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    llm_multimodal::jpeg_turbo::enforce_policy(policy).map_err(|error| {
        anyhow::anyhow!("{error} (the gauge smg_mm_turbojpeg_available reads 0 for this process)")
    })?;
    if policy == llm_multimodal::jpeg_turbo::TurboJpegPolicy::Require {
        SAID.get_or_init(|| {
            tracing::info!("libjpeg-turbo is required by the TurboJPEG policy and was loaded");
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::body::to_bytes;
    use http::StatusCode;
    use llm_multimodal::{hub::HubFetchError, MediaConnectorError, Modality, MultiModalError};

    use super::{
        config_unavailable_response, encoder_budget::ItemOverEncoderBudget,
        invalid_multimodal_request, processing_failed_message, processing_failed_response,
        record_decoder_availability, report_jpeg_decoder, Response,
    };
    use crate::{middleware::test_support::json_of, routers::common::retry::NonRetryable};

    async fn status_and_error(response: Response) -> (u16, serde_json::Value) {
        let status = response.status().as_u16();
        (status, json_of(response).await)
    }

    /// A configuration file the model's hub failed to serve is a 502 under its
    /// own code with a generic message: the hub's answer and its address stay
    /// out of the body, and the response carries no non-retryable mark, since
    /// the next request resolves the checkpoint again. Any other failure is
    /// not this answer.
    #[tokio::test]
    async fn a_processor_config_the_hub_failed_to_serve_answers_a_502_without_the_cause() {
        let failed = anyhow::Error::from(HubFetchError {
            model: "org/name".to_string(),
            file: "preprocessor_config.json",
            cause: "request error: HTTP status server error (503 Service Unavailable) for url \
                    (http://hub.internal/org/name/resolve/main/preprocessor_config.json)"
                .to_string(),
        })
        .context("Failed to resolve model config directory for 'org/name'");
        let response = processing_failed_response(&failed);
        assert!(
            response.extensions().get::<NonRetryable>().is_none(),
            "a hub failure is retried by the next request"
        );
        let (status, json) = status_and_error(response).await;
        assert_eq!(status, 502);
        assert_eq!(json["error"]["code"], "multimodal_config_unavailable");
        let message = json["error"]["message"].as_str().unwrap();
        assert!(
            !message.contains("hub.internal") && !message.contains("503"),
            "the cause stays out of the body: {message}"
        );
        assert!(
            message.contains("configuration could not be fetched"),
            "the body says what was not fetched: {message}"
        );

        assert!(config_unavailable_response(&anyhow::anyhow!("decode failed")).is_none());
    }

    /// An item the engine could never place answers under a code of its own,
    /// a 400 like every other processing failure, with the item's size and
    /// the budget in the message; a failure of another kind keeps its code.
    #[tokio::test]
    async fn an_item_over_the_engines_budget_answers_under_its_own_code() {
        let over = anyhow::Error::new(ItemOverEncoderBudget {
            modality: Modality::Video,
            position: 1,
            count: 1,
            feature_tokens: 192_001,
            sampled_frames: Some(900),
            budget: 192_000,
        })
        .context("Failed to process the plan");
        let (status, json) = status_and_error(processing_failed_response(&over)).await;
        assert_eq!(status, 400);
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "multimodal_item_too_large");
        let message = json["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("192001 feature tokens (900 sampled frames)")
                && message.contains("192000-token encoder budget"),
            "{message}"
        );

        let (status, json) = status_and_error(processing_failed_response(&anyhow::anyhow!(
            "decode failed"
        )))
        .await;
        assert_eq!(status, 400);
        assert_eq!(json["error"]["code"], "multimodal_processing_failed");
    }

    /// A fetch that ran out of its budget is reported with its URL and the
    /// budget, under the route's sentence, without the "failed to finalize"
    /// context frame around it.
    #[test]
    fn a_processing_failure_names_its_cause() {
        let error = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::Timeout {
            url: "https://media.example/clip.mp4".to_string(),
            budget: Duration::from_secs(30),
        }))
        .context("Failed to finalize multimodal tracker");
        assert_eq!(
            processing_failed_message(&error),
            "Multimodal processing failed: media fetch of https://media.example/clip.mp4 timed \
             out after 30s"
        );
    }

    /// The message is the route's sentence and the pipeline's verdict on the
    /// request: the gateway's stage frames around the verdict are the log's
    /// (the chain keeps them), and the verdict's own source is not repeated
    /// after it. The tracker's own typed error under the frames reads its text
    /// alone; a failure with neither keeps its whole chain, whose frames say
    /// what failed.
    #[test]
    fn a_processing_failures_message_leaves_the_stage_frames_to_the_log() {
        let undecodable =
            image::ImageError::Unsupported(image::error::UnsupportedError::from_format_and_kind(
                image::error::ImageFormatHint::Unknown,
                image::error::UnsupportedErrorKind::Format(image::error::ImageFormatHint::Unknown),
            ));
        let decoder_said = undecodable.to_string();
        let error = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::Image(
            undecodable,
        )))
        .context("Failed to finalize multimodal tracker")
        .context("Failed to process the plan");
        assert_eq!(
            processing_failed_message(&error),
            format!("Multimodal processing failed: image decode error: {decoder_said}")
        );
        assert!(
            format!("{error:#}").contains("Failed to finalize multimodal tracker: "),
            "the chain keeps the frames for the log: {error:#}"
        );

        let validation = anyhow::Error::new(MultiModalError::Validation(
            "placeholder 3 has no part".to_string(),
        ))
        .context("Failed to finalize multimodal tracker");
        assert_eq!(
            processing_failed_message(&validation),
            "Multimodal processing failed: tracker validation error: placeholder 3 has no part"
        );

        let config_read = anyhow::anyhow!("No such file or directory (os error 2)")
            .context("Failed to read config.json at /models/m/config.json");
        assert_eq!(
            processing_failed_message(&config_read),
            "Multimodal processing failed: Failed to read config.json at /models/m/config.json: \
             No such file or directory (os error 2)"
        );
        let ranges = anyhow::anyhow!("feature range 0 has zero length")
            .context("Invalid explicit feature ranges in image replacement 2");
        assert_eq!(
            processing_failed_message(&ranges),
            "Multimodal processing failed: Invalid explicit feature ranges in image replacement 2: \
             feature range 0 has zero length"
        );
    }

    /// The status, the `x-smg-error-code` header and the JSON body of an answer.
    async fn answer(response: Response) -> (StatusCode, String, serde_json::Value) {
        let status = response.status();
        let header = response
            .headers()
            .get("x-smg-error-code")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let bytes = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("the answer's body");
        let body = serde_json::from_slice(&bytes).expect("a JSON envelope");
        (status, header, body)
    }

    /// Media that could not be prepared is the request's to fix: a 400 under
    /// the media request code, with the cause after the route's lead.
    #[tokio::test]
    async fn invalid_multimodal_input_maps_to_bad_request() {
        let (status, code, body) =
            answer(invalid_multimodal_request("unsupported audio modality")).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(code, "invalid_multimodal_request");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert_eq!(
            body["error"]["message"],
            "Invalid multimodal request: unsupported audio modality"
        );
    }

    /// A fetch that ran out of its budget and an origin that answered 5xx are
    /// the media host's trouble for now, a 502 `server_error` under
    /// `media_fetch_failed`; a host that cannot be resolved is the request's
    /// own input, a 400 `invalid_request_error` under the same code; a
    /// malformed data URL, a parameter out of range and a host the policy
    /// refuses are the request's own as sent, 400s under
    /// `invalid_multimodal_request`; a clip
    /// over the decoded-size cap and an item over the input cap are 400s
    /// under `media_too_large`; a scratch file that failed on this side and
    /// a preprocessing task that died are 500 `server_error`s; a failure with
    /// no connector verdict keeps the pipeline's own 400 code. The message is
    /// the route's sentence and the cause in every case.
    #[tokio::test]
    async fn a_pipeline_failure_is_answered_by_the_connectors_verdict() {
        let finalize = |media: MediaConnectorError| {
            anyhow::Error::new(MultiModalError::Media(media))
                .context("Failed to finalize multimodal tracker")
        };
        let origin_answered = |status: u16| {
            let response = reqwest::Response::from(
                http::Response::builder()
                    .status(status)
                    .body("")
                    .expect("a response"),
            );
            finalize(MediaConnectorError::Http(
                response
                    .error_for_status()
                    .expect_err("a 5xx is the fetch's error"),
            ))
        };
        let died = tokio::task::spawn_blocking(|| panic!("the preprocessor died"))
            .await
            .expect_err("the task panicked");
        let died = anyhow::Error::new(died).context("Preprocessing task panicked");
        let cases = [
            (
                finalize(MediaConnectorError::Timeout {
                    url: "https://media.example/video_251mb.mp4".to_string(),
                    budget: Duration::from_secs(30),
                }),
                StatusCode::BAD_GATEWAY,
                "media_fetch_failed",
                "server_error",
            ),
            (
                origin_answered(503),
                StatusCode::BAD_GATEWAY,
                "media_fetch_failed",
                "server_error",
            ),
            (
                origin_answered(404),
                StatusCode::BAD_REQUEST,
                "media_fetch_failed",
                "invalid_request_error",
            ),
            (
                finalize(MediaConnectorError::UnresolvableHost {
                    host: "media.invalid".to_string(),
                    source: std::io::Error::from(std::io::ErrorKind::NotFound),
                }),
                StatusCode::BAD_REQUEST,
                "media_fetch_failed",
                "invalid_request_error",
            ),
            (
                finalize(MediaConnectorError::DataUrl(
                    "missing comma in data url".to_string(),
                )),
                StatusCode::BAD_REQUEST,
                "invalid_multimodal_request",
                "invalid_request_error",
            ),
            (
                finalize(MediaConnectorError::InvalidMaxLongSidePixel {
                    value: 0,
                    factor: 28,
                }),
                StatusCode::BAD_REQUEST,
                "invalid_multimodal_request",
                "invalid_request_error",
            ),
            (
                finalize(MediaConnectorError::DisallowedAddress {
                    host: "127.0.0.1".to_string(),
                    address: std::net::IpAddr::from([127, 0, 0, 1]),
                    class: "loopback",
                }),
                StatusCode::BAD_REQUEST,
                "invalid_multimodal_request",
                "invalid_request_error",
            ),
            (
                finalize(MediaConnectorError::DecodedTooLarge {
                    media: "video RGB",
                    bytes: 2_208_384_000,
                    limit: 1_073_741_824,
                    knob: "SMG_VIDEO_MAX_DECODED_BYTES",
                }),
                StatusCode::BAD_REQUEST,
                "media_too_large",
                "invalid_request_error",
            ),
            (
                finalize(MediaConnectorError::PayloadTooLarge {
                    media: "video",
                    limit: 268_435_456,
                }),
                StatusCode::BAD_REQUEST,
                "media_too_large",
                "invalid_request_error",
            ),
            (
                finalize(MediaConnectorError::DecodeIo(std::io::Error::other(
                    "No space left on device (os error 28)",
                ))),
                StatusCode::INTERNAL_SERVER_ERROR,
                "multimodal_processing_failed",
                "server_error",
            ),
            (
                died,
                StatusCode::INTERNAL_SERVER_ERROR,
                "multimodal_processing_failed",
                "server_error",
            ),
            (
                anyhow::Error::new(MultiModalError::UnsupportedContent("image_embeds"))
                    .context("Failed to push content part"),
                StatusCode::BAD_REQUEST,
                "multimodal_processing_failed",
                "invalid_request_error",
            ),
            (
                anyhow::anyhow!("Multimodal not supported for model: m"),
                StatusCode::BAD_REQUEST,
                "multimodal_processing_failed",
                "invalid_request_error",
            ),
        ];
        for (error, status, code, kind) in cases {
            let message = processing_failed_message(&error);
            let (answered, header, body) = answer(processing_failed_response(&error)).await;
            assert_eq!(answered, status, "{message}: {body}");
            assert_eq!(header, code, "{message}");
            assert_eq!(body["error"]["code"], code, "{message}: {body}");
            assert_eq!(body["error"]["type"], kind, "{message}: {body}");
            assert_eq!(body["error"]["message"], message, "{body}");
        }
    }

    /// A video part on a process without a video decoder is the media
    /// stage's own refusal: a 400 `invalid_multimodal_request` whose message
    /// names what is missing, never a server error.
    #[tokio::test]
    async fn a_video_part_without_a_decoder_is_the_media_stages_refusal() {
        let refused = anyhow::Error::new(super::process::VideoDecoderUnavailable {
            why: "ffmpeg executable not found; install ffmpeg to decode video_url inputs",
        });
        let (status, header, body) = answer(processing_failed_response(&refused)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(header, "invalid_multimodal_request");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert_eq!(
            body["error"]["message"],
            "Invalid multimodal request: video input cannot be decoded by this process: ffmpeg \
             executable not found; install ffmpeg to decode video_url inputs",
            "{body}"
        );
    }

    /// The media host's trouble is answered as the host's (a 502) and is
    /// terminal for the retry layer: another attempt, on this worker or
    /// another, would fetch from the same host; the request's own fault stays
    /// a 400, terminal by its status.
    #[test]
    fn a_media_hosts_failure_is_terminal_for_the_retry_layer() {
        use crate::routers::common::retry::is_retryable_response;

        let timed_out = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::Timeout {
            url: "https://media.example/clip.mp4".to_string(),
            budget: Duration::from_secs(30),
        }))
        .context("Failed to finalize multimodal tracker");
        let response = processing_failed_response(&timed_out);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(
            !is_retryable_response(&response),
            "a media host's failure is not replayed"
        );

        let wrong = anyhow::Error::new(MultiModalError::Media(
            MediaConnectorError::UnsupportedScheme("ftp".to_string()),
        ))
        .context("Failed to finalize multimodal tracker");
        let response = processing_failed_response(&wrong);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(!is_retryable_response(&response));
    }

    /// The decoder availability gauges land in the recorder in force when the
    /// report runs: the gateway installs its recorder first (`server.rs`),
    /// so `smg_mm_turbojpeg_available` and `smg_mm_webp_available` are on
    /// /metrics under every policy; a report made before the recorder would
    /// set them on the no-op one, once, and they would never appear.
    #[test]
    fn the_decoder_availability_gauges_land_in_the_installed_recorder() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let (available, webp) =
            ::metrics::with_local_recorder(&recorder, record_decoder_availability);
        let rendered = handle.render();
        let reads = |name: &str, value: bool| {
            rendered.lines().any(|line| {
                line.starts_with(name) && line.ends_with(if value { " 1" } else { " 0" })
            })
        };
        assert!(reads("smg_mm_turbojpeg_available", available), "{rendered}");
        assert!(reads("smg_mm_webp_available", webp), "{rendered}");
    }

    /// The report sets the gauges on every call, the log lines once: a report
    /// made before the gateway's recorder exists (a caller building the
    /// components early) does not leave the start-up's own report with
    /// nothing to set.
    #[test]
    fn a_second_report_still_sets_the_gauges_on_the_recorder_in_force() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        // The first report, on whatever recorder is in force here: the once
        // block is consumed by it (or was, by another test of this process).
        report_jpeg_decoder();

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        ::metrics::with_local_recorder(&recorder, report_jpeg_decoder);
        let rendered = handle.render();
        for gauge in ["smg_mm_turbojpeg_available", "smg_mm_webp_available"] {
            assert!(
                rendered.lines().any(|line| line.starts_with(gauge)
                    && (line.ends_with(" 1") || line.ends_with(" 0"))),
                "{gauge} is set by the second report:\n{rendered}"
            );
        }
    }

    /// The verdict the processing arm writes its WARN on: the connector's
    /// client verdict (also under a context), an item over the encoder's
    /// budget and a video part without a decoder are the request's own
    /// media; a failure with no verdict in its chain is not.
    #[test]
    fn only_an_explicit_verdict_makes_a_failure_the_requests_media() {
        let refused = anyhow::Error::new(MediaConnectorError::InvalidUrl("x".to_string()))
            .context("Failed to finalize multimodal tracker");
        assert!(super::is_request_media_fault(&refused));
        assert!(super::is_request_media_fault(&anyhow::Error::new(
            super::process::VideoDecoderUnavailable { why: "no decoder" }
        )));
        assert!(super::is_request_media_fault(&anyhow::Error::new(
            ItemOverEncoderBudget {
                modality: Modality::Image,
                position: 1,
                count: 1,
                feature_tokens: 2,
                sampled_frames: None,
                budget: 1,
            }
        )));
        assert!(!super::is_request_media_fault(&anyhow::anyhow!(
            "Image preprocessing failed"
        )));
    }
}
