//! MiniMax-M3 vision processor.
//!
//! Despite the `image_grid_pinpoints` and `process_image_mode: "dynamic_res"`
//! keys in its config — which read as LLaVA-NeXT tiling — MiniMax-M3 preprocesses
//! images the Qwen2-VL way: a Qwen-style `smart_resize` onto a
//! `patch_size * merge_size` grid, patchified into a flat
//! `[total_patches, channels * temporal_patch_size * patch_size^2]` tensor
//! alongside an `image_grid_thw` triple. vLLM's M3 vision tower consumes exactly
//! that layout, so this processor wraps the shared [`QwenVLProcessorBase`],
//! supplies M3's own parameters, and raises images below M3's short-side floor
//! before delegating.
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
//! - min short side: 112 px (images below it are raised first; video frames are not);
//!   past roughly 36:1 the raised image overshoots max_pixels, so the grid is its uniform
//!   scale-down and the short side ends below 112 again
//! - normalization: CLIP mean/std
//!
//! The bounds differ from Qwen2-VL's (200,704 / 1,003,520), so M3 cannot simply
//! reuse the Qwen2-VL processor's defaults.

use std::ops::Deref;

use image::{imageops::FilterType, DynamicImage};

use super::{
    qwen2_vl::{CLIP_MEAN, CLIP_STD},
    qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode},
};
use crate::{
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::{pil_to_filter, resize, resize_bicubic_pil, TransformError},
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

/// Short side floor in pixels, four patch factors; smaller images are scaled up to it first.
const MIN_SHORT_SIDE: u32 = 112;

/// The base's aspect-ratio guard (`smart_resize` in qwen_vl_base.rs), which a staged image must
/// satisfy too; keep the two in step.
const MAX_ASPECT_RATIO: f64 = 200.0;

/// How far over the pixel budget a shrunk staging image lands, so the base still scales it down.
const STAGING_OVERSHOOT: f64 = 1.25;

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
    ///
    /// A present but malformed `img_token_compression_config` value fails
    /// here, at construction, rather than silently falling back to the
    /// defaults and hiding the real cause until the first request.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Result<Self, TransformError> {
        Self::new().layered_over(config)
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

    /// Dimensions with the short side raised to [`MIN_SHORT_SIDE`]; `None` if at it or zero.
    fn raised_dimensions(width: u32, height: u32) -> Option<(u32, u32)> {
        let short = width.min(height);
        if short == 0 || short >= MIN_SHORT_SIDE {
            return None;
        }
        let long = (f64::from(width.max(height)) * f64::from(MIN_SHORT_SIDE) / f64::from(short))
            .round() as u32;
        Some(if width <= height {
            (MIN_SHORT_SIDE, long)
        } else {
            (long, MIN_SHORT_SIDE)
        })
    }

    /// Dimensions to hand the base for an image below the floor; `None` if it is at the floor.
    ///
    /// Within the pixel budget that is the raised image itself. Above it the
    /// base's own target is staged when the base would hand it back unchanged
    /// (one resample, at most the budget). Otherwise, for a very thin image or
    /// a lowered budget, the raised image is shrunk to just over the budget,
    /// which trades about a percent of grid fidelity for the bound.
    fn staging_dimensions(
        &self,
        width: u32,
        height: u32,
    ) -> Result<Option<(u32, u32)>, TransformError> {
        let Some((raised_w, raised_h)) = Self::raised_dimensions(width, height) else {
            return Ok(None);
        };
        let (target_h, target_w) = self
            .inner
            .smart_resize(raised_h as usize, raised_w as usize)?;
        let budget = self.inner.max_pixels();
        if raised_w as usize * raised_h as usize <= budget {
            return Ok(Some((raised_w, raised_h)));
        }
        let target_fits = target_w * target_h <= budget
            && aspect_ratio(target_w, target_h) <= MAX_ASPECT_RATIO
            && self.inner.smart_resize(target_h, target_w)? == (target_h, target_w);
        if target_fits {
            return Ok(Some((target_w as u32, target_h as u32)));
        }
        let shrink = (f64::from(raised_w) * f64::from(raised_h)
            / (budget as f64 * STAGING_OVERSHOOT))
            .sqrt();
        // Round the short side up and the long side down so the ratio stays inside the guard.
        let shrunk = |side: u32, short: bool| {
            let scaled = f64::from(side) / shrink;
            if short { scaled.ceil() } else { scaled.floor() }.max(1.0) as u32
        };
        let (w, h) = if raised_w <= raised_h {
            (shrunk(raised_w, true), shrunk(raised_h, false))
        } else {
            (shrunk(raised_w, false), shrunk(raised_h, true))
        };
        let over_budget = w as usize * h as usize > budget;
        if over_budget && aspect_ratio(w as usize, h as usize) <= MAX_ASPECT_RATIO {
            Ok(Some((w, h)))
        } else {
            Ok(Some((raised_w, raised_h)))
        }
    }

    /// Stage every image below the floor, or `None` when none needs it so the batch stays borrowed.
    ///
    /// Otherwise the whole batch is copied, not just the staged images, so the base gets one slice.
    fn raise_images(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<Option<Vec<DynamicImage>>, TransformError> {
        let below_floor =
            |image: &DynamicImage| Self::raised_dimensions(image.width(), image.height()).is_some();
        if !config.do_resize.unwrap_or(true) || !images.iter().any(below_floor) {
            return Ok(None);
        }
        // Vet every image first so an off-contract input is rejected before any pixel work.
        for image in images {
            self.inner
                .smart_resize(image.height() as usize, image.width() as usize)?;
        }
        // The base's grid stage picks its kernel the same way, so both stages use one filter.
        let filter = pil_to_filter(config.resampling.or(Some(3)));
        let mut staged = Vec::with_capacity(images.len());
        for image in images {
            staged.push(
                match self.staging_dimensions(image.width(), image.height())? {
                    Some((width, height)) if filter == FilterType::CatmullRom => {
                        resize_bicubic_pil(image, width, height)
                    }
                    Some((width, height)) => resize(image, width, height, filter),
                    None => image.clone(),
                },
            );
        }
        Ok(Some(staged))
    }
}

/// Long side over short side.
fn aspect_ratio(a: usize, b: usize) -> f64 {
    a.max(b) as f64 / a.min(b).max(1) as f64
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
        let layered = self.for_request(config)?;
        let Some(raised) = layered.raise_images(images, config)? else {
            return layered.inner.preprocess(images, config);
        };
        let mut out = layered.inner.preprocess(&raised, config)?;
        // Report the caller's sizes, not the raised ones.
        out.item_sizes = images
            .iter()
            .map(|image| (image.width(), image.height()))
            .collect();
        Ok(out)
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
        let layered = self.for_request(config).unwrap_or_else(|_| self.clone());
        let staged = config
            .do_resize
            .unwrap_or(true)
            .then(|| layered.staging_dimensions(width, height).ok().flatten())
            .flatten();
        let (width, height) = staged.unwrap_or((width, height));
        layered.inner.calculate_num_tokens(width, height, config)
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
    fn short_sides_below_the_floor_are_raised_before_the_patch_grid() {
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        // 400x40 is raised to 1120x112: an 8x80 patch grid, 160 tokens after the 2x2 merge.
        assert_eq!(processor.calculate_num_tokens(400, 40, &config), 160);
        assert_eq!(processor.calculate_num_tokens(40, 400, &config), 160);
        // A short side already at the floor is left alone.
        assert_eq!(processor.calculate_num_tokens(400, 112, &config), 56);
    }

    #[test]
    fn raised_dimensions_raise_only_short_sides_below_the_floor() {
        assert_eq!(
            MiniMaxM3VisionProcessor::raised_dimensions(400, 40),
            Some((1120, 112))
        );
        assert_eq!(
            MiniMaxM3VisionProcessor::raised_dimensions(40, 400),
            Some((112, 1120))
        );
        assert_eq!(
            MiniMaxM3VisionProcessor::raised_dimensions(50, 50),
            Some((112, 112))
        );
        assert_eq!(
            MiniMaxM3VisionProcessor::raised_dimensions(112, 20),
            Some((627, 112))
        );
        assert_eq!(MiniMaxM3VisionProcessor::raised_dimensions(112, 400), None);
        assert_eq!(MiniMaxM3VisionProcessor::raised_dimensions(0, 40), None);
    }

    #[test]
    fn tokens_do_not_decrease_as_the_short_side_shrinks_within_the_pixel_budget() {
        // At a fixed long side, shrinking the short side below the floor raises
        // it by a larger factor, so the token count must not decrease while the
        // raised image still fits max_pixels (the verifier's 10_16 band).
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        let tokens: Vec<usize> = [90, 40, 20]
            .into_iter()
            .map(|short| processor.calculate_num_tokens(400, short, &config))
            .collect();
        assert_eq!(tokens, vec![72, 160, 320]);
    }

    #[test]
    fn past_the_pixel_budget_the_uniform_clamp_wins() {
        // Beyond roughly 36:1 the raised image overshoots max_pixels, so the
        // base scales it down uniformly and the short side ends below 112
        // again, as the reference processor does; monotonicity is not kept.
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        let tokens: Vec<usize> = [20, 10, 5]
            .into_iter()
            .map(|short| processor.calculate_num_tokens(400, short, &config))
            .collect();
        assert_eq!(tokens, vec![320, 453, 428]);
    }

    fn gradient(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y * 6 % 256) as u8, ((x + y) % 256) as u8])
        }))
    }

    #[test]
    fn the_raise_uses_the_requests_resample_filter() {
        // With `resample` = nearest, the wrapper's stage must produce the
        // pixels the base would get from a nearest-neighbour raise, and they
        // must differ from the bicubic ones.
        let processor = MiniMaxM3VisionProcessor::new();
        let image = gradient(400, 40);
        let mut nearest = m3_config();
        nearest.resampling = Some(0);
        let mut bicubic = m3_config();
        bicubic.resampling = Some(3);

        let via_wrapper = processor
            .preprocess(std::slice::from_ref(&image), &nearest)
            .unwrap();
        let staged = resize(&image, 1120, 112, FilterType::Nearest);
        let via_base = processor.inner.preprocess(&[staged], &nearest).unwrap();
        assert_eq!(via_wrapper.encoder_input, via_base.encoder_input);

        let via_bicubic = processor.preprocess(&[image], &bicubic).unwrap();
        assert_ne!(via_wrapper.encoder_input, via_bicubic.encoder_input);
    }

    #[test]
    fn do_resize_false_skips_the_raise() {
        // The caller asked for no resizing, so the floor does not apply and
        // the base rejects the off-grid buffer exactly as it did before.
        let processor = MiniMaxM3VisionProcessor::new();
        let mut config = m3_config();
        config.do_resize = Some(false);
        assert!(processor
            .preprocess(&[DynamicImage::new_rgb8(400, 40)], &config)
            .is_err());
        assert_eq!(processor.calculate_num_tokens(400, 40, &config), 14);
    }

    #[test]
    fn staged_images_stay_near_the_pixel_budget() {
        let processor = MiniMaxM3VisionProcessor::new();
        let cap = (processor.max_pixels() as f64 * 1.3) as usize;
        for (width, height) in [
            (200, 1),
            (150, 1),
            (1000, 6),
            (4000, 24),
            (400, 10),
            (400, 5),
        ] {
            let (w, h) = processor
                .staging_dimensions(width, height)
                .unwrap()
                .expect("below the floor");
            assert!(
                w as usize * h as usize <= cap,
                "{width}x{height} staged at {w}x{h}"
            );
            assert!(
                aspect_ratio(w as usize, h as usize) <= MAX_ASPECT_RATIO,
                "{width}x{height}"
            );
        }
        // Within the budget the raised image itself is staged.
        assert_eq!(
            processor.staging_dimensions(400, 40).unwrap(),
            Some((1120, 112))
        );
        // Over it, the base's own target is staged once its ratio is acceptable.
        assert_eq!(
            processor.staging_dimensions(400, 10).unwrap(),
            Some((4228, 84))
        );
    }

    #[test]
    fn preprocess_resizes_a_flat_image_onto_the_raised_grid() {
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        let out = processor
            .preprocess(&[DynamicImage::new_rgb8(400, 40)], &config)
            .expect("flat image preprocesses");
        assert_eq!(out.feature_token_counts, vec![160]);
        let crate::ModelSpecificValue::IntTensor { data, .. } =
            &out.model_specific["image_grid_thw"]
        else {
            panic!("image_grid_thw is an int tensor");
        };
        assert_eq!(data, &[1, 8, 80]);
    }

    #[test]
    fn token_count_matches_preprocess_for_raised_images() {
        // The placeholder count and the produced grid come from different
        // entry points; they must agree, including where the raised long side
        // rounds to the patch factor (112x20 -> 627x112 -> 616x112).
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        for (width, height) in [(400, 40), (40, 400), (112, 20), (50, 50), (300, 80)] {
            let out = processor
                .preprocess(&[DynamicImage::new_rgb8(width, height)], &config)
                .unwrap();
            assert_eq!(
                out.feature_token_counts,
                vec![processor.calculate_num_tokens(width, height, &config)],
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn thin_images_within_the_aspect_limit_still_preprocess() {
        // The base's 200:1 aspect guard must see the raised dimensions, whose
        // ratio equals the original's, never an intermediate grid.
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        for (width, height) in [(672, 4), (1008, 6), (4000, 24), (150, 1), (200, 1)] {
            let out = processor
                .preprocess(&[DynamicImage::new_rgb8(width, height)], &config)
                .unwrap_or_else(|err| panic!("{width}x{height}: {err}"));
            assert_eq!(
                out.feature_token_counts,
                vec![processor.calculate_num_tokens(width, height, &config)],
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn off_contract_aspect_ratios_are_still_rejected_with_the_callers_dimensions() {
        // Raising keeps the aspect ratio, so the base's guard fires for the
        // same inputs as before and names the dimensions the caller sent.
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        for (width, height) in [(201, 1), (2, 500), (30_000, 100)] {
            let err = processor
                .preprocess(&[DynamicImage::new_rgb8(width, height)], &config)
                .expect_err("aspect ratio above 200:1 is rejected");
            let TransformError::InvalidShape { actual, .. } = err else {
                panic!("{width}x{height}: {err}");
            };
            assert_eq!(actual, vec![height as usize, width as usize]);
        }
    }

    #[test]
    fn a_mixed_batch_with_an_off_contract_image_is_rejected_as_a_whole() {
        // The oversized image is at the floor and never raised; it must still
        // be vetted before the batch is copied.
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        let images = [
            DynamicImage::new_rgb8(112, 30_000),
            DynamicImage::new_rgb8(50, 50),
        ];
        let err = processor
            .preprocess(&images, &config)
            .expect_err("rejected");
        let TransformError::InvalidShape { actual, .. } = err else {
            panic!("{err}");
        };
        assert_eq!(actual, vec![30_000, 112]);
    }

    #[test]
    fn verifier_rule_b_cases_get_their_raised_grids() {
        // m3_image_tests 10_15: the five (width, height) cases and the token
        // count each produces once the short side is raised to 112.
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        let tokens: Vec<usize> = [(400, 40), (300, 80), (112, 20), (40, 400), (80, 300)]
            .into_iter()
            .map(|(width, height)| processor.calculate_num_tokens(width, height, &config))
            .collect();
        assert_eq!(tokens, vec![160, 60, 88, 160, 60]);
    }

    #[test]
    fn token_count_matches_preprocess_under_pixel_overrides() {
        let processor = MiniMaxM3VisionProcessor::new();
        let mut config = m3_config();
        config.max_pixels = Some(100_352);
        config.min_pixels = Some(50_176);
        for (width, height) in [(400, 40), (50, 50), (672, 4), (300, 80), (400, 3)] {
            let out = processor
                .preprocess(&[DynamicImage::new_rgb8(width, height)], &config)
                .unwrap();
            assert_eq!(
                out.feature_token_counts,
                vec![processor.calculate_num_tokens(width, height, &config)],
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn a_target_over_a_lowered_budget_is_not_staged() {
        // With max_pixels = 100,352 the base's target for 400x3 is 3640x28,
        // over the budget, so the base would scale it again; the shrunk raise
        // is staged instead and the grid is pinned.
        let processor = MiniMaxM3VisionProcessor::new();
        let mut config = m3_config();
        config.max_pixels = Some(100_352);
        let layered = processor.layered_over(&config).unwrap();
        assert_eq!(
            layered.staging_dimensions(400, 3).unwrap(),
            Some((4089, 31))
        );
        assert_eq!(processor.calculate_num_tokens(400, 3, &config), 129);
    }

    #[test]
    fn thin_band_grids_are_pinned() {
        // Past 144:1 the raise is shrunk before staging; the grids are within
        // a percent of the full raise's (293 and 339 tokens).
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        assert_eq!(processor.calculate_num_tokens(150, 1, &config), 292);
        assert_eq!(processor.calculate_num_tokens(200, 1, &config), 336);
    }

    #[test]
    fn mixed_batches_keep_their_order() {
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        let images = [
            DynamicImage::new_rgb8(224, 224),
            DynamicImage::new_rgb8(400, 40),
            DynamicImage::new_rgb8(224, 224),
        ];
        let out = processor.preprocess(&images, &config).unwrap();
        assert_eq!(out.feature_token_counts, vec![64, 160, 64]);
    }

    #[test]
    fn item_sizes_report_the_callers_dimensions() {
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        let images = [
            DynamicImage::new_rgb8(400, 40),
            DynamicImage::new_rgb8(224, 224),
        ];
        let out = processor.preprocess(&images, &config).unwrap();
        assert_eq!(out.item_sizes, vec![(400, 40), (224, 224)]);
    }

    #[test]
    fn video_frames_follow_the_base_unchanged() {
        // The floor is an image rule here; clips take the base's own path.
        let processor = MiniMaxM3VisionProcessor::new();
        let config = m3_config();
        let frames = vec![DynamicImage::new_rgb8(400, 40); 2];
        let out = processor.preprocess_video(&frames, &config).unwrap();
        let base = processor.inner.preprocess_video(&frames, &config).unwrap();
        assert_eq!(out.feature_token_counts, base.feature_token_counts);
        assert_eq!(out.feature_token_counts, vec![14]);
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
