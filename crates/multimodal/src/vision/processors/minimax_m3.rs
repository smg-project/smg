//! MiniMax-M3 vision processor.
//!
//! Despite the `image_grid_pinpoints` and `process_image_mode: "dynamic_res"`
//! keys in its config — which read as LLaVA-NeXT tiling — MiniMax-M3 preprocesses
//! images the Qwen2-VL way: a Qwen-style `smart_resize` onto a
//! `patch_size * merge_size` grid, patchified into a flat
//! `[total_patches, channels * temporal_patch_size * patch_size^2]` tensor
//! alongside an `image_grid_thw` triple. The engine's M3 vision tower consumes exactly
//! that layout, so this processor wraps the shared [`QwenVLProcessorBase`] and
//! supplies M3's own parameters. Images are resized once, where the checkpoint's
//! image processor plans it: its `smart_resize` from the source size (each side
//! rounded to the grid and floored at one factor, then the pixel bounds), so a
//! square up to 70 px lands on a 4x4 grid and a 2000x40 image keeps its rounded
//! long side (28x1988); nothing raises a short side first.
//!
//! Router and Rust-servicer images use `minimax_resize` to choose aligned dimensions
//! from the source header and resize directly at decode. Their processor carries
//! `RESIZED_KEY`, skipping the legacy image pixel budget.
//! SGLang native workers explicitly select day-0 nearest-grid geometry and Pillow
//! byte resizing. Other processors retain the defaults below.
//!
//! # MiniMax-M3 parameters
//!
//! - patch_size: 14
//! - merge_size: 2 (`img_token_compression_config.spatial_merge_size`)
//! - temporal_patch_size: 2 (`img_token_compression_config.temporal_patch_size`)
//! - factor: 28 (patch_size * merge_size)
//! - min_pixels: 3,136 (4 * 28 * 28)
//! - max_pixels: 451,584 (576 * 28 * 28) — matches `image_seq_length: 576`
//! - video max_pixels: 602,112 (768 * 28 * 28), bounding each frame rather than
//!   the sampled volume, as the reference video processor applies `smart_resize`
//!   to the frame size. A request that names a `max_long_side_pixel` tier has
//!   its frames capped to that long side upstream, and the tier's square becomes
//!   the frame budget instead: shrinking those frames again to the default would
//!   make the 1008 and 2016 tiers produce the same tokens for 16:9 sources.
//! - normalization: CLIP mean/std
//!
//! The bounds differ from Qwen2-VL's (200,704 / 1,003,520), so M3 cannot simply
//! reuse the Qwen2-VL processor's defaults.

use std::{borrow::Cow, ops::Deref};

#[path = "minimax_sglang_video.rs"]
mod sglang_video;

use image::DynamicImage;

use super::{
    qwen2_vl::{CLIP_MEAN, CLIP_STD},
    qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode},
};
use crate::{
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VideoClipOptions, VisionPreProcessor},
        transforms::TransformError,
    },
    EncoderDtype,
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

/// Default maximum pixels per video frame (768 * 28 * 28 = 602,112), not per sampled volume.
pub const DEFAULT_VIDEO_MAX_PIXELS: usize = 768 * 28 * 28;

/// The config block holding M3's merge parameters.
const COMPRESSION_CONFIG_KEY: &str = "img_token_compression_config";

/// MiniMax-M3 image/video processor.
#[derive(Clone)]
pub struct MiniMaxM3VisionProcessor {
    inner: QwenVLProcessorBase,
    /// Whether the config set `max_pixels`, which then bounds video frames
    /// in place of a request's `max_long_side_pixel` tier.
    explicit_max_pixels: bool,
    /// Opt in to SGLang's day-0 geometry and Pillow byte resize for images and videos.
    sglang_resize: bool,
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
                video_resize_mode: QwenVideoResizeMode::PerFrame,
                temporal_patch_size,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "minimax_m3",
            })
            .allow_video_dimensions_below_factor()
            .raise_rounded_sides_to_factor(),
            explicit_max_pixels: false,
            sglang_resize: false,
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

    /// The `temporal_patch_size` the config declares: a positive flat key, else the compression block.
    fn declared_temporal_patch_size(
        config: &PreProcessorConfig,
    ) -> Result<Option<usize>, TransformError> {
        let nested = Self::compression_usize(config, "temporal_patch_size")?;
        Ok(config
            .temporal_patch_size
            .filter(|&size| size > 0)
            .or(nested))
    }

    /// Frames per temporal patch this processor pairs under `config`: the flat
    /// key, else the compression block, else the default. A present but
    /// unusable block value is the configuration error the construction path
    /// reports, so a prompt's stamps are never paired by a default that the
    /// preprocessing of the same config refuses.
    pub fn temporal_patch_size_from(config: &PreProcessorConfig) -> Result<usize, TransformError> {
        Ok(Self::declared_temporal_patch_size(config)?.unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE))
    }

    /// Build a processor from a preprocessor config, falling back to M3's
    /// defaults for anything the config does not specify.
    ///
    /// A present but malformed `img_token_compression_config` value fails
    /// here, at construction, rather than silently falling back to the
    /// defaults and hiding the real cause until the first request.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Result<Self, TransformError> {
        Self::new().layered_over(config)
    }

    /// Layer a config over this processor's settings.
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
        let temporal_patch_size = Self::declared_temporal_patch_size(config)?
            .unwrap_or_else(|| self.inner.temporal_patch_size());
        let source_resized = config
            .get_extra::<bool>(crate::minimax_resize::RESIZED_KEY)
            .unwrap_or(false);
        if source_resized
            && config
                .get_patch_size(self.inner.patch_size())
                .checked_mul(merge_size)
                != Some(28)
        {
            return Err(TransformError::ShapeError(
                "M3 source resizing requires patch_size * merge_size = 28".into(),
            ));
        }
        // Source planning already aligned and checked the fixed image budget.
        let max_pixels = if source_resized {
            crate::minimax_resize::IMAGE_MAX_TOTAL_PIXELS as usize
        } else {
            config.max_pixels.unwrap_or_else(|| self.inner.max_pixels())
        };
        let min_pixels = if source_resized {
            28 * 28
        } else {
            config.min_pixels.unwrap_or_else(|| self.inner.min_pixels())
        };
        // Videos retain the explicit legacy area limit, even when image source
        // planning is enabled. Do not floor a lower video limit at the default.
        let video_max_pixels = config
            .max_pixels
            .unwrap_or_else(|| self.inner.video_max_pixels());

        let mut layered = Self::build(
            config.get_patch_size(self.inner.patch_size()),
            merge_size,
            temporal_patch_size,
            min_pixels,
            max_pixels,
            video_max_pixels,
        );
        layered.inner = layered.inner.with_preprocessor_config(config);
        layered.sglang_resize = config
            .get_extra::<bool>(crate::minimax_resize::SGLANG_RESIZE_KEY)
            .unwrap_or(false);
        layered.explicit_max_pixels = self.explicit_max_pixels || config.max_pixels.is_some();
        Ok(layered)
    }

    /// This processor for one clip: a `max_long_side_pixel` tier on the
    /// request is the caller's frame budget (see the module docs), unless
    /// `max_pixels` was set explicitly. The budget is the square of the long
    /// side, so a frame already capped to that side is never shrunk again.
    fn for_clip(&self, clip: &VideoClipOptions) -> Cow<'_, Self> {
        match clip.max_long_side_pixel {
            Some(tier) if !self.explicit_max_pixels => {
                let tier = tier as usize;
                let budget = tier.saturating_mul(tier).max(self.inner.min_pixels());
                let mut processor = self.clone();
                processor.inner = processor.inner.with_video_max_pixels(budget);
                Cow::Owned(processor)
            }
            _ => Cow::Borrowed(self),
        }
    }
}

impl Deref for MiniMaxM3VisionProcessor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for MiniMaxM3VisionProcessor {
    /// Each image contributes independent BF16 patch rows and metadata.
    fn supports_bf16_batch_cache(&self) -> bool {
        true
    }

    fn emits_pixel_bytes(&self) -> bool {
        true
    }

    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.inner.preprocess(images)
    }

    /// Write multi-image BF16 batches in one pass. Single images fall back to
    /// neutral U8 pixels, which the gateway converts when filling its BF16 cache.
    fn preprocess_wire(
        &self,
        images: &[&DynamicImage],
        dtype: EncoderDtype,
    ) -> Option<Result<PreprocessedEncoderInputs, TransformError>> {
        if dtype != EncoderDtype::BFloat16 || images.len() < 2 {
            return None;
        }
        self.inner.preprocess_wire(images, dtype)
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if self.sglang_resize {
            let owned: Vec<_> = frames.iter().map(DynamicImage::to_rgb8).collect();
            let borrowed: Vec<_> = owned
                .iter()
                .map(|frame| RgbFrameRef {
                    width: frame.width(),
                    height: frame.height(),
                    data: frame.as_raw(),
                })
                .collect();
            return self.preprocess_sglang_video(&borrowed, clip);
        }
        self.for_clip(clip).inner.preprocess_video(frames, clip)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if self.sglang_resize {
            return self.preprocess_sglang_video(frames, clip);
        }
        self.for_clip(clip).inner.preprocess_video_rgb(frames, clip)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        self.inner.calculate_num_tokens(width, height)
    }

    /// The rows cover each resized RGB image once per temporal copy.
    fn image_input_elements(&self, width: u32, height: u32) -> Option<usize> {
        self.inner.image_input_elements(width, height)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        self.inner.get_processed_size()
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

    /// The fused BF16 path preserves pixels, geometry, and metadata under
    /// checkpoint/default settings and the gateway's decode-time resizing.
    #[test]
    fn bf16_wire_matches_neutral_pixels() {
        let images = [(56, 84), (700, 672), (112, 56)].map(|(w, h)| {
            DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([x as u8, y as u8, (x + y) as u8])
            }))
        });
        let mut resized = m3_config();
        resized.extra.insert(
            crate::minimax_resize::RESIZED_KEY.to_owned(),
            serde_json::json!(true),
        );
        let mut unnormalized = m3_config();
        unnormalized.do_normalize = Some(false);
        for config in [
            PreProcessorConfig::default(),
            m3_config(),
            resized,
            unnormalized,
        ] {
            let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
            for refs in [
                vec![&images[0], &images[1]],
                vec![&images[2], &images[0], &images[1], &images[2]],
            ] {
                let neutral = processor.preprocess(&refs).unwrap();
                let elements: usize = refs
                    .iter()
                    .map(|image| {
                        processor
                            .image_input_elements(image.width(), image.height())
                            .unwrap()
                    })
                    .sum();
                assert_eq!(elements, neutral.encoder_input.len());
                let wire = processor
                    .preprocess_wire(&refs, EncoderDtype::BFloat16)
                    .unwrap()
                    .unwrap();
                assert_eq!(wire.encoder_input_shape(), neutral.encoder_input_shape());
                assert_eq!(
                    wire.encoder_input.write_as(EncoderDtype::BFloat16),
                    neutral.encoder_input.write_as(EncoderDtype::BFloat16)
                );
                assert_eq!(wire.feature_token_counts, neutral.feature_token_counts);
                assert_eq!(wire.item_sizes, neutral.item_sizes);
                assert_eq!(wire.model_specific.len(), neutral.model_specific.len());
                for (key, value) in &neutral.model_specific {
                    assert_eq!(
                        format!("{value:?}"),
                        format!("{:?}", wire.model_specific[key])
                    );
                }
            }
            for refs in [vec![], vec![&images[0]]] {
                assert!(processor
                    .preprocess_wire(&refs, EncoderDtype::BFloat16)
                    .is_none());
            }
            let invalid = DynamicImage::new_rgb8(0, 0);
            assert!(processor
                .preprocess_wire(&[&invalid, &images[0]], EncoderDtype::BFloat16)
                .unwrap()
                .is_err());
            for dtype in [
                EncoderDtype::Float32,
                EncoderDtype::Float16,
                EncoderDtype::Uint8,
            ] {
                assert!(processor
                    .preprocess_wire(&[&images[0], &images[1]], dtype)
                    .is_none());
            }
            assert!(processor.supports_bf16_batch_cache());
            assert!(!processor.supports_per_image_preprocessing());
        }
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
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config())
            .expect("checkpoint config is valid");

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
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config())
            .expect("checkpoint config is valid");
        assert_eq!(processor.min_pixels(), 3136);
        assert_eq!(processor.max_pixels(), 451_584);
    }

    #[test]
    fn explicit_flat_keys_win_over_the_nested_block() {
        let mut config = m3_config();
        config.merge_size = Some(4);
        config.temporal_patch_size = Some(1);

        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config)
            .expect("checkpoint config is valid");
        assert_eq!(processor.merge_size(), 4);
        assert_eq!(processor.temporal_patch_size(), 1);
    }

    #[test]
    fn token_count_follows_the_merged_grid() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();

        // A 448x448 image is 32x32 patches at patch_size 14, which is
        // 16x16 = 256 tokens after the 2x2 merge.
        assert_eq!(processor.calculate_num_tokens(448, 448).unwrap(), 256);
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
    fn a_video_tier_sets_the_frame_budget() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let clip = VideoClipOptions {
            sample_fps: 2.0,
            max_long_side_pixel: Some(2016),
        };

        let layered = processor.for_clip(&clip);
        assert_eq!(layered.video_max_pixels(), 2016 * 2016);
        // The tier speaks for video frames only.
        assert_eq!(layered.max_pixels(), DEFAULT_MAX_PIXELS);

        // An explicit max_pixels still wins over the tier.
        let mut config = m3_config();
        config.max_pixels = Some(100_352);
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
        let layered = processor.for_clip(&clip);
        assert_eq!(layered.video_max_pixels(), 100_352);
    }

    #[test]
    fn a_tier_budget_never_drops_below_the_frame_minimum() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let clip = VideoClipOptions {
            sample_fps: 2.0,
            max_long_side_pixel: Some(28),
        };

        let layered = processor.for_clip(&clip);
        assert_eq!(layered.video_max_pixels(), layered.min_pixels());
    }

    #[test]
    fn video_tiers_change_the_token_count() {
        use crate::vision::processor::VisionPreProcessor;

        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let tokens = |width: u32, height: u32, tier: Option<u32>| {
            let clip = VideoClipOptions {
                sample_fps: 2.0,
                max_long_side_pixel: tier,
            };
            let frames = vec![DynamicImage::new_rgb8(width, height); 4];
            processor
                .preprocess_video(&frames, &clip)
                .unwrap()
                .feature_token_counts[0]
        };

        // A 16:9 source, capped upstream to each tier's long side.
        let low = tokens(504, 284, Some(504));
        let mid = tokens(1008, 567, Some(1008));
        let high = tokens(2016, 1134, Some(2016));
        assert!(
            low < mid && mid < high,
            "tiers must order the token count: {low} < {mid} < {high}"
        );

        // Without a tier the reference per-frame budget (768 * 28 * 28) applies,
        // and a 2016-wide frame is shrunk onto the same grid as a 1008-wide one.
        assert_eq!(tokens(2016, 1134, None), tokens(1008, 567, None));
        assert_eq!(tokens(1008, 567, None), mid);
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
    fn temporal_patch_size_from_follows_the_layering_precedence() {
        // The flat key, then the checkpoint block, then the default.
        let mut config = m3_config();
        assert_eq!(
            MiniMaxM3VisionProcessor::temporal_patch_size_from(&config).unwrap(),
            2
        );
        config.temporal_patch_size = Some(4);
        assert_eq!(
            MiniMaxM3VisionProcessor::temporal_patch_size_from(&config).unwrap(),
            4
        );
        assert_eq!(
            MiniMaxM3VisionProcessor::temporal_patch_size_from(&PreProcessorConfig::default())
                .unwrap(),
            DEFAULT_TEMPORAL_PATCH_SIZE
        );
    }

    #[test]
    fn temporal_patch_size_from_rejects_unusable_block_values() {
        // A zero flat key defers to the block; a zero or malformed block value
        // is the configuration error the construction path reports, not the
        // default.
        let mut config = m3_config();
        config.temporal_patch_size = Some(0);
        assert_eq!(
            MiniMaxM3VisionProcessor::temporal_patch_size_from(&config).unwrap(),
            2
        );
        for bad in ["0", "-1", "2.5", "\"two\"", "null"] {
            let raw =
                format!(r#"{{"img_token_compression_config": {{"temporal_patch_size": {bad}}}}}"#);
            let config: PreProcessorConfig = serde_json::from_str(&raw).unwrap();
            let error = MiniMaxM3VisionProcessor::temporal_patch_size_from(&config)
                .unwrap_err()
                .to_string();
            assert!(error.contains("temporal_patch_size"), "{error}");
            let constructed = MiniMaxM3VisionProcessor::from_preprocessor_config(&config)
                .err()
                .unwrap()
                .to_string();
            assert_eq!(
                error, constructed,
                "temporal_patch_size {bad}: one reader, one error"
            );
        }
    }

    #[test]
    fn layering_pairs_frames_by_the_temporal_patch_size_the_lookup_reports() {
        let mut flat = m3_config();
        flat.temporal_patch_size = Some(4);
        let mut zero_flat = m3_config();
        zero_flat.temporal_patch_size = Some(0);
        let nested: PreProcessorConfig =
            serde_json::from_str(r#"{"img_token_compression_config": {"temporal_patch_size": 3}}"#)
                .unwrap();

        for config in [
            m3_config(),
            flat,
            zero_flat,
            nested,
            PreProcessorConfig::default(),
        ] {
            let layered = MiniMaxM3VisionProcessor::new()
                .layered_over(&config)
                .unwrap();
            assert_eq!(
                layered.temporal_patch_size(),
                MiniMaxM3VisionProcessor::temporal_patch_size_from(&config).unwrap()
            );
        }
    }

    #[test]
    fn from_preprocessor_config_rejects_a_malformed_compression_block() {
        // Construction must fail loudly rather than fall back to the defaults
        // and hide the real cause until the first request.
        let config: PreProcessorConfig = serde_json::from_str(
            r#"{"patch_size": 14,
                 "img_token_compression_config": {"spatial_merge_size": "two"}}"#,
        )
        .unwrap();
        assert!(MiniMaxM3VisionProcessor::from_preprocessor_config(&config).is_err());
    }

    #[test]
    fn video_preprocessing_is_supported() {
        use crate::vision::processor::VisionPreProcessor;

        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        // Two frames so the temporal patch pairing has something to pair.
        let frames = vec![
            DynamicImage::new_rgb8(224, 224),
            DynamicImage::new_rgb8(224, 224),
        ];

        // The shared Qwen base implements the video path; M3 must delegate to
        // it rather than falling through to the "unsupported" default.
        let out = processor
            .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
            .expect("M3 supports video preprocessing");
        assert!(!out.feature_token_counts.is_empty());
    }

    #[test]
    fn narrow_video_matches_minimax_reference() {
        use crate::vision::processor::ModelSpecificValue;

        // Generated by scripts/generate_minimax_narrow_video_golden.py using
        // the pinned official processor. Check every normalized patch value,
        // including channel/temporal ordering, as well as the grid and tokens.
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/golden/minimax_narrow_video.json"
        ))
        .unwrap();
        let processor = MiniMaxM3VisionProcessor::new();
        for case in fixture["cases"].as_array().unwrap() {
            let width = case["width"].as_u64().unwrap() as u32;
            let height = case["height"].as_u64().unwrap() as u32;
            let frames = fixture["colors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|color| {
                    DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                        width,
                        height,
                        image::Rgb(std::array::from_fn(|i| color[i].as_u64().unwrap() as u8)),
                    ))
                })
                .collect::<Vec<_>>();
            let output = processor
                .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
                .unwrap();
            let shape = case["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(output.encoder_input.shape(), shape);
            assert_eq!(output.feature_token_counts, vec![shape[0] / 4]);
            let expected_grid = case["grid"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>();
            assert!(matches!(
                output.model_specific.get("video_grid_thw"),
                Some(ModelSpecificValue::IntTensor { data, shape })
                    if data == &expected_grid && shape == &[1, 3]
            ));
            for (index, &value) in output.encoder_input.as_f32().iter().enumerate() {
                let expected = case["channel_frame_values"][(index % 1176) / 196]
                    .as_f64()
                    .unwrap() as f32;
                assert!(
                    (value - expected).abs() < 1e-6,
                    "{width}x{height} patch value {index}: {value} != {expected}"
                );
            }
        }
    }

    #[test]
    fn video_zero_dimensions_are_rejected() {
        let processor = MiniMaxM3VisionProcessor::new();
        for (width, height) in [(0, 100), (100, 0), (0, 0)] {
            let frames = vec![DynamicImage::new_rgb8(width, height); 2];
            assert!(processor
                .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
                .is_err());
        }
    }

    /// `image_grid_thw` of one preprocessed image.
    fn image_grid_thw(out: &PreprocessedEncoderInputs) -> Vec<i64> {
        let crate::ModelSpecificValue::IntTensor { data, .. } =
            &out.model_specific["image_grid_thw"]
        else {
            panic!("image_grid_thw is an int tensor");
        };
        data.clone()
    }

    #[test]
    fn tiny_and_extreme_aspect_images_take_the_checkpoint_processors_grid() {
        // The grids and token counts of the checkpoint's image processor, whose
        // `smart_resize` plans one resize from the source size: each side rounded
        // to the 28 px grid and floored at one factor, then the 3,136 / 451,584
        // pixel bounds. Nothing raises a short side first, so a square up to 70 px
        // is a 4x4 grid and a 2000x40 image keeps its rounded long side (28x1988).
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        for (width, height, grid, tokens) in [
            (1, 1, [1, 4, 4], 4),
            (2, 2, [1, 4, 4], 4),
            (7, 5, [1, 4, 6], 6),
            (16, 16, [1, 4, 4], 4),
            (64, 64, [1, 4, 4], 4),
            (128, 96, [1, 6, 10], 15),
            (2000, 40, [1, 2, 142], 71),
            (40, 2000, [1, 142, 2], 71),
            // Ordinary sizes do not move.
            (1024, 768, [1, 40, 54], 540),
            (1920, 1080, [1, 36, 64], 576),
        ] {
            let out = processor
                .preprocess(&[&DynamicImage::new_rgb8(width, height)])
                .unwrap_or_else(|err| panic!("{width}x{height}: {err}"));
            assert_eq!(image_grid_thw(&out), grid, "{width}x{height} grid");
            assert_eq!(
                out.feature_token_counts,
                vec![tokens],
                "{width}x{height} tokens"
            );
            assert_eq!(
                processor.calculate_num_tokens(width, height).unwrap(),
                tokens,
                "{width}x{height} count"
            );
            assert_eq!(
                out.item_sizes,
                vec![(width, height)],
                "{width}x{height} size"
            );
        }
    }

    #[test]
    fn thin_images_within_the_aspect_limit_preprocess_and_count_alike() {
        // A side that rounds to nothing is floored at one factor while the long
        // side keeps its rounded length; the count and the tensor agree.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        for (width, height, tokens) in [
            (672, 4, 24),
            (1008, 6, 36),
            (4000, 24, 143),
            (150, 1, 5),
            (200, 1, 7),
        ] {
            let out = processor
                .preprocess(&[&DynamicImage::new_rgb8(width, height)])
                .unwrap_or_else(|err| panic!("{width}x{height}: {err}"));
            assert_eq!(out.feature_token_counts, vec![tokens], "{width}x{height}");
            assert_eq!(
                processor.calculate_num_tokens(width, height).unwrap(),
                tokens,
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn off_contract_aspect_ratios_are_rejected_with_the_callers_dimensions() {
        // The base's 200:1 guard reads the source size and names it.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        for (width, height) in [(201, 1), (2, 500), (30_000, 100)] {
            let err = processor
                .preprocess(&[&DynamicImage::new_rgb8(width, height)])
                .expect_err("aspect ratio above 200:1 is rejected");
            let TransformError::InvalidShape { actual, .. } = err else {
                panic!("{width}x{height}: {err}");
            };
            assert_eq!(actual, vec![height as usize, width as usize]);
        }
    }

    #[test]
    fn a_mixed_batch_with_an_off_contract_image_is_rejected_as_a_whole() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let images = [
            DynamicImage::new_rgb8(112, 30_000),
            DynamicImage::new_rgb8(50, 50),
        ];
        let err = processor
            .preprocess(&images.iter().collect::<Vec<_>>())
            .expect_err("rejected");
        let TransformError::InvalidShape { actual, .. } = err else {
            panic!("{err}");
        };
        assert_eq!(actual, vec![30_000, 112]);
    }

    #[test]
    fn token_count_matches_preprocess_under_pixel_overrides() {
        let mut config = m3_config();
        config.max_pixels = Some(100_352);
        config.min_pixels = Some(50_176);
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
        for (width, height) in [(400, 40), (50, 50), (672, 4), (300, 80), (400, 3)] {
            let out = processor
                .preprocess(&[&DynamicImage::new_rgb8(width, height)])
                .unwrap();
            assert_eq!(
                out.feature_token_counts,
                vec![processor.calculate_num_tokens(width, height).unwrap()],
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn mixed_batches_keep_their_order() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let images = [
            DynamicImage::new_rgb8(224, 224),
            DynamicImage::new_rgb8(400, 40),
            DynamicImage::new_rgb8(224, 224),
        ];
        let out = processor
            .preprocess(&images.iter().collect::<Vec<_>>())
            .unwrap();
        assert_eq!(out.feature_token_counts, vec![64, 14, 64]);
    }

    #[test]
    fn item_sizes_report_the_callers_dimensions() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let images = [
            DynamicImage::new_rgb8(400, 40),
            DynamicImage::new_rgb8(224, 224),
        ];
        let out = processor
            .preprocess(&images.iter().collect::<Vec<_>>())
            .unwrap();
        assert_eq!(out.item_sizes, vec![(400, 40), (224, 224)]);
    }

    #[test]
    fn images_and_video_frames_take_the_same_grid_rule() {
        // One rule for both: a 400x40 image and a clip of 400x40 frames land on
        // the base's 2x28 grid, 14 tokens per temporal patch.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let image = processor
            .preprocess(&[&DynamicImage::new_rgb8(400, 40)])
            .unwrap();
        assert_eq!(image.feature_token_counts, vec![14]);
        let frames = vec![DynamicImage::new_rgb8(400, 40); 2];
        let clip = processor
            .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
            .unwrap();
        assert_eq!(clip.feature_token_counts, vec![14]);
    }

    fn video_grid_thw(out: &PreprocessedEncoderInputs) -> Vec<i64> {
        let crate::ModelSpecificValue::IntTensor { data, .. } =
            &out.model_specific["video_grid_thw"]
        else {
            panic!("video_grid_thw is an int tensor");
        };
        data.clone()
    }

    #[test]
    fn the_video_budget_bounds_each_frame_not_the_sampled_volume() {
        // 720p lands on 560x1008 (720 tokens per pair) at any frame count; a volume budget gives 392x728 (364) for two frames.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        assert_eq!(processor.video_resize_mode(), QwenVideoResizeMode::PerFrame);
        assert_eq!(processor.video_max_pixels(), 602_112);
        assert_eq!(
            processor.smart_resize_video(2, 720, 1280).unwrap(),
            (560, 1008)
        );
        assert_eq!(
            processor.smart_resize_video(16, 720, 1280).unwrap(),
            (560, 1008)
        );
        let out = processor
            .preprocess_video(
                &vec![DynamicImage::new_rgb8(1280, 720); 2],
                &VideoClipOptions::sampled_at(2.0),
            )
            .unwrap();
        assert_eq!(video_grid_thw(&out), vec![1, 40, 72]);
        assert_eq!(out.feature_token_counts, vec![720]);

        let out = processor
            .preprocess_video(
                &vec![DynamicImage::new_rgb8(1280, 720); 16],
                &VideoClipOptions::sampled_at(2.0),
            )
            .unwrap();
        assert_eq!(video_grid_thw(&out), vec![8, 40, 72]);
        assert_eq!(out.feature_token_counts, vec![5760]);
    }

    #[test]
    fn large_images_are_bounded_by_max_pixels() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();

        // Well past the bound: the count must clamp to image_seq_length.
        let tokens = processor.calculate_num_tokens(4096, 4096).unwrap();
        assert!(
            tokens <= 576,
            "expected at most image_seq_length (576) tokens, got {tokens}"
        );
    }

    #[test]
    fn smart_resize_floors_a_side_that_rounds_to_zero_like_the_reference() {
        // MiniMax-M3's reference takes `max(factor, round(side / factor) * factor)`
        // before the pixel-budget test, so a 1x100 image keeps its rounded long
        // side (28x112 = exactly min_pixels, no scale-up); the Qwen rule would
        // scale it up to 28x560.
        let processor = MiniMaxM3VisionProcessor::new();
        assert_eq!(processor.inner.smart_resize(1, 100).unwrap(), (28, 112));
        assert_eq!(processor.inner.smart_resize(100, 1).unwrap(), (112, 28));
    }
}
