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

use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
};

use llm_multimodal::{
    AudioClip, EncoderFieldLayouts, ImageFrame, Modality, PlaceholderRange,
    PreprocessedEncoderInputs, VideoClip,
};
use llm_tokenizer::{Encoding, TokenizerTrait};

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
mod pixel_cache;
mod plan;
mod process;
mod serialize;
mod transport;

pub(crate) use assemble::{
    assemble_multimodal_data, assemble_multimodal_data_after_encode,
    assemble_tokenspeed_for_encode, encode_routing_hashes,
};
pub(crate) use capability::ensure_backend_supports_modalities;
pub(crate) use config::{
    load_image_preprocessor_config, load_video_preprocessor_config, MultimodalComponents,
    MultimodalConfigRegistry, MultimodalModelConfig,
};
pub(crate) use detect::{media_plan_chat, media_plan_messages};
pub(crate) use plan::{
    prepare_placeholder_tokens, resolve_media_part_order, validate_rendered_media_anchors,
    PlaceholderTokens,
};
pub(crate) use process::process_multimodal_plan;
pub(crate) use transport::{init_mm_transport_defaults, mm_rdma_exporter};

/// Whether verbose multimodal timing logs are enabled via `SMG_LOG_MM_TIMING`.
/// Read from the environment once and cached; the flag is not expected to change
/// at runtime, and this is called on every multimodal request.
fn log_mm_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SMG_LOG_MM_TIMING")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
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
        Ok(Self { batches })
    }

    pub(crate) fn batches(&self) -> &[PrecomputedMultimodalIntermediate] {
        &self.batches
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
}
