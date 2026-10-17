//! Multimodal processing integration for gRPC pipeline (chat + messages).
//!
//! Bridges the `llm-multimodal` crate with the gRPC router pipeline, split by
//! processing phase:
//!
//! - [`detect`]: find modalities and extract content parts from chat/messages.
//! - [`config`]: model config-file registry and per-router component bundle.
//! - [`process`]: fetch media → preprocess → expand placeholder tokens →
//!   build the lightweight [`MultimodalIntermediate`].
//! - [`assemble`]: turn the intermediate into backend-specific `MultimodalData`
//!   once the target backend is known (after worker selection).
//! - [`serialize`]: tensor byte/dtype serialization used by assembly.
//! - [`transport`]: SHM-vs-inline transport resolution and `/dev/shm`
//!   namespace verification.
//! - [`refs`]: router-vs-worker processing resolution and the media-reference
//!   payload for workers that process media themselves.

use std::{collections::HashSet, sync::Arc};

use axum::response::Response;
use llm_multimodal::{
    AudioClip, EncoderFieldLayouts, EncoderInputFormat, ImageFrame, Modality, PlaceholderRange,
    PreprocessedEncoderInputs, VideoClip,
};
use llm_tokenizer::{Encoding, TokenizerTrait};

use crate::routers::{error, grpc::proto_wrapper::VLLM_MROPE_GRID_KEYS};

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
mod inflight;
mod item_limits;
mod pixel_cache;
mod plan;
mod process;
mod refs;
mod serialize;
mod settings;
mod transparency;
mod transport;
pub mod worker;

pub(crate) use assemble::{
    assemble_multimodal_data, assemble_multimodal_data_after_encode,
    assemble_tokenspeed_for_encode, encode_routing_hashes,
};
pub(crate) use capability::{
    ensure_backend_supports_modalities, ensure_model_accepts_media, fleet_reads_format,
    request_needs_vision, worker_language_model_only, SUPPORTS_VISION_LABEL,
};
pub(crate) use config::{
    load_image_preprocessor_config, load_video_preprocessor_config, MultimodalComponents,
    MultimodalConfigRegistry, MultimodalModelConfig,
};
pub(crate) use detect::{media_plan_chat, media_plan_messages};
pub(crate) use inflight::{
    reserve_multimodal_inflight, InflightPermit, UNIT_BYTES as INFLIGHT_UNIT_BYTES,
};
pub(crate) use item_limits::engine_item_limits;
pub(crate) use plan::{
    anchor_rendered_markers, prepare_placeholder_tokens, resolve_media_part_order,
    validate_rendered_media_anchors, MediaPlan, PlaceholderTokens,
};
pub(crate) use process::process_multimodal_plan;
pub(crate) use refs::{
    assemble_media_refs, ensure_selection_supports_media_refs, media_refs_pipeline_pin,
    resolve_mm_processing, worker_accepts_media_refs, worker_runs_gateway_pipeline, MmProcessing,
    MmRefsError, GATEWAY_PIPELINE_PROCESSOR,
};
pub(crate) use serialize::WireMismatch;
pub(crate) use settings::{init_mm_settings, mm_settings, MultimodalSettings};
pub(crate) use transparency::engine_transparency_policy;
pub(crate) use transport::{
    init_mm_transport_defaults, mm_rdma_exporter, mm_shared_image_wire_dtype_over,
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

/// Whether verbose multimodal timing logs are enabled (`--log-mm-timing` /
/// `SMG_LOG_MM_TIMING`). Resolved once at startup; called on every multimodal
/// request.
/// The client-facing message for a media pipeline failure: the whole error
/// chain, so the cause (the URL whose fetch ran out of its budget, the clip
/// the decoder refused) reaches the caller, not only the outermost context.
pub(crate) fn processing_failed_message(error: &anyhow::Error) -> String {
    format!("Multimodal processing failed: {error:#}")
}

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
        })
    }

    pub(crate) fn with_encoder_input_format(mut self, format: EncoderInputFormat) -> Self {
        self.encoder_input_format = format;
        self
    }

    pub(crate) fn encoder_input_format(&self) -> EncoderInputFormat {
        self.encoder_input_format
    }

    pub(crate) fn batches(&self) -> &[PrecomputedMultimodalIntermediate] {
        &self.batches
    }

    /// Whether a batch carries the grid tensors a disaggregated decode leg
    /// derives its positions from ([`VLLM_MROPE_GRID_KEYS`]): such a leg
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
/// read: the one call [`report_jpeg_decoder`] makes once per process, kept
/// apart so a test can run it against a recorder of its own. The gateway
/// installs its Prometheus recorder before the report (`server.rs`): a gauge
/// set before that lands in the no-op recorder and never reaches /metrics.
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
    ONCE.get_or_init(|| {
        let (available, _webp) = record_decoder_availability();
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

    use llm_multimodal::{MediaConnectorError, MultiModalError};

    use super::{processing_failed_message, record_decoder_availability};

    /// A fetch that ran out of its budget is reported with its URL and the
    /// budget, not as the bare "failed to finalize" context around it.
    #[test]
    fn a_processing_failure_names_its_cause() {
        let error = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::Timeout {
            url: "https://media.example/clip.mp4".to_string(),
            budget: Duration::from_secs(30),
        }))
        .context("Failed to finalize multimodal tracker");
        assert_eq!(
            processing_failed_message(&error),
            "Multimodal processing failed: Failed to finalize multimodal tracker: media fetch of \
             https://media.example/clip.mp4 timed out after 30s"
        );
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
}
