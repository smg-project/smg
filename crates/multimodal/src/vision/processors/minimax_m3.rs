//! MiniMax-M3 vision processor.
//!
//! Despite the `image_grid_pinpoints` and `process_image_mode: "dynamic_res"`
//! keys in its config — which read as LLaVA-NeXT tiling — MiniMax-M3 preprocesses
//! images the Qwen2-VL way: a Qwen-style `smart_resize` onto a
//! `patch_size * merge_size` grid, patchified into a flat
//! `[total_patches, channels * temporal_patch_size * patch_size^2]` tensor
//! alongside an `image_grid_thw` triple. vLLM's M3 vision tower consumes exactly
//! that layout, so this processor wraps the shared [`QwenVLProcessorBase`] and
//! only supplies M3's own parameters.
//!
//! # MiniMax-M3 parameters
//!
//! - patch_size: 14
//! - merge_size: 2 (`img_token_compression_config.spatial_merge_size`)
//! - temporal_patch_size: 2 (`img_token_compression_config.temporal_patch_size`)
//! - factor: 28 (patch_size * merge_size)
//! - min_pixels: 3,136 (4 * 28 * 28)
//! - max_pixels: 451,584 (576 * 28 * 28) — matches `image_seq_length: 576`
//! - video max_pixels: 602,112 (768 * 28 * 28)
//! - normalization: CLIP mean/std
//!
//! The bounds differ from Qwen2-VL's (200,704 / 1,003,520), so M3 cannot simply
//! reuse the Qwen2-VL processor's defaults.

use std::ops::Deref;

use image::DynamicImage;

use super::{
    qwen2_vl::{CLIP_MEAN, CLIP_STD},
    qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode},
};
use crate::{
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::TransformError,
    },
};

/// Default patch size.
pub const DEFAULT_PATCH_SIZE: usize = 14;

/// Default spatial merge size (2x2 patch merge before the projector).
pub const DEFAULT_MERGE_SIZE: usize = 2;

/// Default temporal patch size (video frames are padded to a multiple of this).
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// Default minimum pixels (4 * 28 * 28 = 3,136).
pub const DEFAULT_MIN_PIXELS: usize = 4 * 28 * 28;

/// Default maximum pixels for images (576 * 28 * 28 = 451,584).
///
/// 576 is the model's `image_seq_length`: at the bound, an image occupies
/// exactly `image_seq_length` tokens after the 2x2 merge (2,304 patches before
/// it).
pub const DEFAULT_MAX_PIXELS: usize = 576 * 28 * 28;

/// Default maximum pixels per video frame (768 * 28 * 28 = 602,112).
pub const DEFAULT_VIDEO_MAX_PIXELS: usize = 768 * 28 * 28;

/// The config block holding M3's merge parameters.
const COMPRESSION_CONFIG_KEY: &str = "img_token_compression_config";

/// MiniMax-M3 image/video processor.
#[derive(Clone)]
pub struct MiniMaxM3VisionProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for MiniMaxM3VisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniMaxM3VisionProcessor {
    /// Create a processor with MiniMax-M3's default parameters.
    pub fn new() -> Self {
        Self::build(
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_TEMPORAL_PATCH_SIZE,
            DEFAULT_MIN_PIXELS,
            DEFAULT_MAX_PIXELS,
            DEFAULT_VIDEO_MAX_PIXELS,
        )
    }

    /// Create a processor with custom settings.
    pub fn with_config(
        patch_size: usize,
        merge_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        temporal_patch_size: usize,
    ) -> Self {
        // The caller's `max_pixels` governs video too; flooring it at the
        // video default would ignore a deliberate reduction.
        Self::build(
            patch_size,
            merge_size,
            temporal_patch_size,
            min_pixels,
            max_pixels,
            max_pixels,
        )
    }

    fn build(
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        video_max_pixels: usize,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels,
                max_pixels,
                video_min_pixels: min_pixels,
                video_max_pixels,
                video_resize_mode: QwenVideoResizeMode::TotalVolume,
                temporal_patch_size,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "minimax_m3",
            }),
        }
    }

    /// Read a `usize` out of M3's `img_token_compression_config` block.
    ///
    /// M3 nests its merge parameters there rather than exposing the flat
    /// `merge_size` / `temporal_patch_size` keys Qwen models use, so they land
    /// in `PreProcessorConfig::extra` instead of the typed fields.
    ///
    /// Returns `Ok(None)` only when the key is genuinely absent. A present but
    /// malformed value (non-integer, or zero) is an error rather than a silent
    /// fallback: defaulting there would run every request with tensor geometry
    /// that disagrees with the checkpoint.
    fn compression_usize(
        config: &PreProcessorConfig,
        key: &str,
    ) -> Result<Option<usize>, TransformError> {
        let Some(block) = config.extra.get(COMPRESSION_CONFIG_KEY) else {
            return Ok(None);
        };
        let Some(value) = block.get(key) else {
            return Ok(None);
        };
        match value.as_u64() {
            Some(parsed) if parsed > 0 => Ok(Some(parsed as usize)),
            _ => Err(TransformError::ShapeError(format!(
                "minimax_m3: {COMPRESSION_CONFIG_KEY}.{key} must be a positive integer, got {value}"
            ))),
        }
    }

    /// Build a processor from a preprocessor config, falling back to M3's
    /// defaults for anything the config does not specify.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self::new()
            .layered_over(config)
            .unwrap_or_else(|_| Self::new())
    }

    /// Layer a request's config over this processor's settings.
    ///
    /// Values the config does not specify keep whatever this processor was
    /// built with, so settings supplied through [`Self::with_config`] survive
    /// into `preprocess` and `calculate_num_tokens` instead of being reset to
    /// the checkpoint defaults.
    fn layered_over(&self, config: &PreProcessorConfig) -> Result<Self, TransformError> {
        let merge_size = config
            .merge_size
            .or(Self::compression_usize(config, "spatial_merge_size")?)
            .unwrap_or_else(|| self.inner.merge_size());
        let temporal_patch_size = config
            .temporal_patch_size
            .or(Self::compression_usize(config, "temporal_patch_size")?)
            .unwrap_or_else(|| self.inner.temporal_patch_size());
        let max_pixels = config.max_pixels.unwrap_or_else(|| self.inner.max_pixels());
        let min_pixels = config.min_pixels.unwrap_or_else(|| self.inner.min_pixels());
        // Track an explicit `max_pixels` in both directions: flooring the video
        // budget at the default would leave video six times an image's budget
        // for an operator who lowered `max_pixels` to bound encoder memory.
        let video_max_pixels = config
            .max_pixels
            .unwrap_or_else(|| self.inner.video_max_pixels());

        Ok(Self::build(
            config.get_patch_size(self.inner.patch_size()),
            merge_size,
            temporal_patch_size,
            min_pixels,
            max_pixels,
            video_max_pixels,
        ))
    }

    /// Rebuild for one request so per-request config overrides take effect.
    ///
    /// M3's structural parameters live in `extra`, which
    /// `has_structural_overrides` does not account for, so this always layers
    /// rather than checking for overrides first.
    fn for_request(&self, config: &PreProcessorConfig) -> Result<Self, TransformError> {
        self.layered_over(config)
    }
}

impl Deref for MiniMaxM3VisionProcessor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for MiniMaxM3VisionProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.for_request(config)?.inner.preprocess(images, config)
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.for_request(config)?
            .inner
            .preprocess_video(frames, config)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.for_request(config)?
            .inner
            .preprocess_video_rgb(frames, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        // Infallible signature: a malformed config surfaces on the preprocess
        // call, so fall back to this processor's own settings here.
        self.for_request(config)
            .unwrap_or_else(|_| self.clone())
            .inner
            .calculate_num_tokens(width, height, config)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        self.inner.get_processed_size(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `img_token_compression_config` block as it appears in the
    /// MiniMax-M3 checkpoint's `preprocessor_config.json`.
    fn m3_config() -> PreProcessorConfig {
        serde_json::from_str(
            r#"{
                "processor_class": "MiniMaxVLProcessor",
                "process_image_mode": "dynamic_res",
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "size": [672, 672],
                "patch_size": 14,
                "img_token_compression_config": {
                    "image_token_compression_threshold": 1.1,
                    "image_token_compression_method": "patch_merge",
                    "max_image_resolution": 1008,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2
                },
                "add_start_end_special_tokens": true
            }"#,
        )
        .expect("checkpoint preprocessor config parses")
    }

    #[test]
    fn defaults_match_the_checkpoint() {
        let processor = MiniMaxM3VisionProcessor::new();
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.temporal_patch_size(), 2);
        assert_eq!(processor.min_pixels(), 3136);
        assert_eq!(processor.max_pixels(), 451_584);
        assert_eq!(processor.model_name(), "minimax_m3");
    }

    #[test]
    fn max_pixels_matches_image_seq_length() {
        // 576 is the model's image_seq_length; at the bound an image is
        // exactly that many patches before the 2x2 merge.
        let processor = MiniMaxM3VisionProcessor::new();
        let factor = processor.patch_size() * processor.merge_size();
        assert_eq!(processor.max_pixels() / (factor * factor), 576);
    }

    #[test]
    fn reads_merge_params_from_the_nested_compression_block() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config());

        // Neither key exists at the top level of M3's config; both must be
        // picked up from img_token_compression_config.
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.temporal_patch_size(), 2);
        assert_eq!(processor.patch_size(), 14);
    }

    #[test]
    fn checkpoint_config_keeps_m3_pixel_bounds() {
        // M3's config carries no min_pixels/max_pixels, so the M3 defaults must
        // survive rather than falling back to Qwen2-VL's much larger bounds.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config());
        assert_eq!(processor.min_pixels(), 3136);
        assert_eq!(processor.max_pixels(), 451_584);
    }

    #[test]
    fn explicit_flat_keys_win_over_the_nested_block() {
        let mut config = m3_config();
        config.merge_size = Some(4);
        config.temporal_patch_size = Some(1);

        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config);
        assert_eq!(processor.merge_size(), 4);
        assert_eq!(processor.temporal_patch_size(), 1);
    }

    #[test]
    fn token_count_follows_the_merged_grid() {
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();

        // A 448x448 image is 32x32 patches at patch_size 14, which is
        // 16x16 = 256 tokens after the 2x2 merge.
        assert_eq!(processor.calculate_num_tokens(448, 448, &config), 256);
    }

    #[test]
    fn with_config_settings_survive_request_layering() {
        // A processor built with explicit settings must keep them when a
        // request config does not override them.
        let processor = MiniMaxM3VisionProcessor::with_config(14, 2, 3136, 200_704, 2);
        let mut config = m3_config();
        // The checkpoint block carries merge/temporal but no pixel bounds.
        config.max_pixels = None;
        config.min_pixels = None;

        let layered = processor.layered_over(&config).unwrap();
        assert_eq!(layered.max_pixels(), 200_704);
        assert_eq!(layered.min_pixels(), 3136);
    }

    #[test]
    fn lowering_max_pixels_also_lowers_the_video_budget() {
        let processor = MiniMaxM3VisionProcessor::new();
        let mut config = m3_config();
        config.max_pixels = Some(100_352);

        let layered = processor.layered_over(&config).unwrap();
        assert_eq!(layered.max_pixels(), 100_352);
        // Must track downwards, not stay floored at the video default.
        assert_eq!(layered.video_max_pixels(), 100_352);
    }

    #[test]
    fn malformed_compression_values_are_rejected() {
        for bad in ["\"two\"", "0", "2.5", "null"] {
            let raw = format!(
                r#"{{"patch_size": 14,
                     "img_token_compression_config": {{"spatial_merge_size": {bad}}}}}"#
            );
            let config: PreProcessorConfig = serde_json::from_str(&raw).unwrap();
            let err = MiniMaxM3VisionProcessor::new().layered_over(&config);
            assert!(
                err.is_err(),
                "a present but malformed spatial_merge_size ({bad}) must fail loudly"
            );
        }
    }

    #[test]
    fn absent_compression_block_uses_defaults() {
        let config: PreProcessorConfig = serde_json::from_str(r#"{"patch_size": 14}"#).unwrap();
        let layered = MiniMaxM3VisionProcessor::new()
            .layered_over(&config)
            .unwrap();
        assert_eq!(layered.merge_size(), DEFAULT_MERGE_SIZE);
        assert_eq!(layered.temporal_patch_size(), DEFAULT_TEMPORAL_PATCH_SIZE);
    }

    #[test]
    fn video_preprocessing_is_supported() {
        use crate::vision::processor::VisionPreProcessor;

        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        // Two frames so the temporal patch pairing has something to pair.
        let frames = vec![
            DynamicImage::new_rgb8(224, 224),
            DynamicImage::new_rgb8(224, 224),
        ];

        // The shared Qwen base implements the video path; M3 must delegate to
        // it rather than falling through to the "unsupported" default.
        let out = processor
            .preprocess_video(&frames, &config)
            .expect("M3 supports video preprocessing");
        assert!(!out.feature_token_counts.is_empty());
    }

    #[test]
    fn large_images_are_bounded_by_max_pixels() {
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();

        // Well past the bound: the count must clamp to image_seq_length.
        let tokens = processor.calculate_num_tokens(4096, 4096, &config);
        assert!(
            tokens <= 576,
            "expected at most image_seq_length (576) tokens, got {tokens}"
        );
    }
}
