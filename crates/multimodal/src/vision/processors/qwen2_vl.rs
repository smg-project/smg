//! Qwen2-VL family image processors.
//!
//! This module provides the Qwen2-VL processor which wraps the shared
//! `QwenVLProcessorBase` with Qwen2-VL specific default parameters.
//!
//! # Key Features
//!
//! - **Smart Resize**: Resizes images to fit within min/max pixel bounds while
//!   preserving aspect ratio and aligning to patch boundaries
//! - **Dynamic Token Count**: Token count depends on actual image dimensions
//! - **image_grid_thw**: Returns (T, H, W) grid dimensions for position encoding
//!
//! # Qwen2-VL Parameters
//!
//! - patch_size: 14
//! - merge_size: 2
//! - factor: 28 (patch_size * merge_size)
//! - normalization: CLIP mean/std

use std::ops::Deref;

use image::DynamicImage;

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode};
use crate::{
    encoder_inputs::EncoderDtype,
    types::{Modality, RgbFrameRef},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VideoClipOptions, VisionPreProcessor},
        transforms::TransformError,
    },
};

/// CLIP normalization mean values used by Qwen2-VL models.
pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization std values used by Qwen2-VL models.
pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Default minimum pixels (256 * 28 * 28 = 200,704)
pub const DEFAULT_MIN_PIXELS: usize = 256 * 28 * 28;

/// Default maximum pixels (1280 * 28 * 28 = 1,003,520)
pub const DEFAULT_MAX_PIXELS: usize = 1280 * 28 * 28;

/// Default patch size
pub const DEFAULT_PATCH_SIZE: usize = 14;

/// Default merge size for token reduction
pub const DEFAULT_MERGE_SIZE: usize = 2;

/// Default temporal patch size (for video frames)
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// The video processor's own minimum pixels per frame, for a config that
/// names none (128 * 28 * 28 = 100,352).
pub const DEFAULT_VIDEO_MIN_PIXELS: usize = 128 * 28 * 28;

/// The video processor's own maximum pixels per frame, for a config that
/// names none (768 * 28 * 28 = 602,112).
pub const DEFAULT_VIDEO_MAX_PIXELS: usize = 768 * 28 * 28;

/// Qwen2-VL image processor.
///
/// This is a thin wrapper around `QwenVLProcessorBase` with Qwen2-VL
/// specific default parameters:
/// - patch_size: 14
/// - merge_size: 2
/// - CLIP normalization mean/std
#[derive(Debug, Clone)]
pub struct Qwen2VLProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for Qwen2VLProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen2VLProcessor {
    /// Create a new Qwen2-VL processor with default settings.
    ///
    /// Defaults:
    /// - patch_size: 14
    /// - merge_size: 2
    /// - min_pixels: 200,704 (256 * 28 * 28)
    /// - max_pixels: 1,003,520 (1280 * 28 * 28)
    /// - temporal_patch_size: 2
    /// - normalization: CLIP mean/std
    pub fn new() -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size: DEFAULT_PATCH_SIZE,
                merge_size: DEFAULT_MERGE_SIZE,
                min_pixels: DEFAULT_MIN_PIXELS,
                max_pixels: DEFAULT_MAX_PIXELS,
                video_min_pixels: DEFAULT_MIN_PIXELS,
                video_max_pixels: DEFAULT_MAX_PIXELS,
                video_resize_mode: QwenVideoResizeMode::TotalVolume,
                temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "qwen2-vl",
            }),
        }
    }

    /// Create a processor with custom settings.
    pub fn with_config(
        patch_size: usize,
        merge_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        temporal_patch_size: usize,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels,
                max_pixels,
                video_min_pixels: min_pixels,
                video_max_pixels: max_pixels,
                video_resize_mode: QwenVideoResizeMode::TotalVolume,
                temporal_patch_size,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "qwen2-vl",
            }),
        }
    }

    /// Create a processor from preprocessor config, resolving its geometry
    /// and transforms once.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            inner: QwenVLProcessorBase::from_configs(
                QwenVLConfig {
                    patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
                    merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
                    min_pixels: config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
                    max_pixels: config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
                    video_min_pixels: config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
                    video_max_pixels: config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
                    video_resize_mode: QwenVideoResizeMode::TotalVolume,
                    temporal_patch_size: config
                        .temporal_patch_size
                        .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
                    mean: CLIP_MEAN,
                    std: CLIP_STD,
                    model_name: "qwen2-vl",
                },
                config,
            ),
        }
    }

    /// The processor for `modality` from the config the checkpoint ships for
    /// it: for a clip the video processor's own pixel bounds, read as that
    /// processor reads them (the `min_pixels` / `max_pixels` of a config
    /// shared with the image processor, which is what a checkpoint shipping
    /// one `preprocessor_config.json` gives both, else its `size` edges, else
    /// that processor's own defaults); for an image the image processor's.
    pub fn from_config_for(config: &PreProcessorConfig, modality: Modality) -> Self {
        if modality == Modality::Video {
            Self::from_video_preprocessor_config(config)
        } else {
            Self::from_preprocessor_config(config)
        }
    }

    /// [`Self::from_preprocessor_config`] with the video bounds the video
    /// processor reads: the config's `min_pixels` / `max_pixels`, else its
    /// `size` edges, else that processor's own defaults; every frame is sized
    /// within them on its own, as that processor sizes it.
    fn from_video_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let video_min_pixels = config
            .min_pixels
            .or_else(|| config.get_shortest_edge())
            .unwrap_or(DEFAULT_VIDEO_MIN_PIXELS);
        let video_max_pixels = config
            .max_pixels
            .or_else(|| config.get_longest_edge())
            .unwrap_or(DEFAULT_VIDEO_MAX_PIXELS);
        Self {
            inner: QwenVLProcessorBase::from_configs(
                QwenVLConfig {
                    patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
                    merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
                    min_pixels: config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
                    max_pixels: config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
                    video_min_pixels,
                    video_max_pixels,
                    // Each frame is sized within the video bounds on its own, as the
                    // family's own video processor sizes it; the clip's total is not capped.
                    video_resize_mode: QwenVideoResizeMode::PerFrame,
                    temporal_patch_size: config
                        .temporal_patch_size
                        .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
                    mean: CLIP_MEAN,
                    std: CLIP_STD,
                    model_name: "qwen2-vl",
                },
                config,
            ),
        }
    }

    /// Get the patch size.
    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }

    /// Get the merge size.
    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }

    /// Get the minimum pixels.
    pub fn min_pixels(&self) -> usize {
        self.inner.min_pixels()
    }

    /// Get the maximum pixels.
    pub fn max_pixels(&self) -> usize {
        self.inner.max_pixels()
    }

    /// Get the temporal patch size.
    pub fn temporal_patch_size(&self) -> usize {
        self.inner.temporal_patch_size()
    }

    /// Get the factor for dimension alignment.
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.inner.get_factor()
    }

    /// Smart resize algorithm for Qwen2-VL.
    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        self.inner.smart_resize(height, width)
    }

    /// Calculate the grid dimensions (T, H, W) for an image.
    pub fn calculate_grid_thw(
        &self,
        height: usize,
        width: usize,
        num_frames: usize,
    ) -> (usize, usize, usize) {
        self.inner.calculate_grid_thw(height, width, num_frames)
    }

    /// Calculate the number of image tokens after merge.
    pub fn calculate_tokens_from_grid(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> usize {
        self.inner
            .calculate_tokens_from_grid(grid_t, grid_h, grid_w)
    }
}

impl Deref for Qwen2VLProcessor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for Qwen2VLProcessor {
    fn supports_per_image_preprocessing(&self) -> bool {
        // The shared Qwen image pipeline plans and writes each image independently.
        true
    }

    /// Each image contributes independent BF16 patch rows and grid metadata.
    fn supports_bf16_batch_cache(&self) -> bool {
        true
    }

    /// Use the same geometry as the shared Qwen image encoder for admission.
    fn image_input_elements(&self, width: u32, height: u32) -> Option<usize> {
        self.inner.image_input_elements(width, height)
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

    fn preprocess_wire(
        &self,
        images: &[&DynamicImage],
        dtype: EncoderDtype,
    ) -> Option<Result<PreprocessedEncoderInputs, TransformError>> {
        self.inner.preprocess_wire(images, dtype)
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.inner.preprocess_video(frames, clip)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.inner.preprocess_video_rgb(frames, clip)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        self.inner.calculate_num_tokens(width, height)
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
    use image::Rgb;

    use super::*;
    use crate::{
        test_support::{assert_two_image_batch, create_test_image},
        vision::{preprocessor_config::PatchSize, processor::ModelSpecificValue},
    };

    #[test]
    fn test_qwen2_vl_processor_default() {
        let processor = Qwen2VLProcessor::new();
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.min_pixels(), DEFAULT_MIN_PIXELS);
        assert_eq!(processor.max_pixels(), DEFAULT_MAX_PIXELS);
        assert_eq!(processor.get_factor(), 28); // 14 * 2
    }

    /// A clip goes through the shared video pipeline: four 56 px frames at
    /// patch 14 and merge 2 are two temporal groups of a 4 x 4 grid, eight
    /// tokens, with the clip's grid, patch count and timing beside the
    /// patches. The video bounds come from the config the way the video
    /// processor reads them: its `min_pixels` / `max_pixels`, else that
    /// processor's own defaults.
    #[test]
    fn video_frames_preprocess_through_the_shared_pipeline() {
        let processor = Qwen2VLProcessor::with_config(14, 2, 56 * 56, 56 * 56 * 4, 2);
        let frames: Vec<DynamicImage> = (0..4).map(|_| DynamicImage::new_rgb8(56, 56)).collect();
        let clip = VideoClipOptions {
            sample_fps: 2.0,
            max_long_side_pixel: None,
        };
        let inputs = processor.preprocess_video(&frames, &clip).expect("a clip");
        assert_eq!(inputs.feature_token_counts, vec![8]);
        for key in [
            "video_grid_thw",
            "patches_per_video",
            "video_second_per_grid",
        ] {
            assert!(inputs.model_specific.contains_key(key), "{key}");
        }
        let config = PreProcessorConfig {
            min_pixels: Some(3136),
            max_pixels: Some(12_845_056),
            ..PreProcessorConfig::default()
        };
        let from_config = Qwen2VLProcessor::from_config_for(&config, Modality::Video);
        assert_eq!(from_config.video_min_pixels(), 3136);
        assert_eq!(from_config.video_max_pixels(), 12_845_056);
        let bare =
            Qwen2VLProcessor::from_config_for(&PreProcessorConfig::default(), Modality::Video);
        assert_eq!(bare.video_min_pixels(), DEFAULT_VIDEO_MIN_PIXELS);
        assert_eq!(bare.video_max_pixels(), DEFAULT_VIDEO_MAX_PIXELS);
        let image = Qwen2VLProcessor::from_config_for(&config, Modality::Image);
        assert_eq!(image.min_pixels(), 3136);
    }

    #[test]
    fn test_smart_resize_within_bounds() {
        let processor = Qwen2VLProcessor::new();

        // Image that's already within bounds
        let (h, w) = processor.smart_resize(500, 500).unwrap();

        // Should be aligned to factor (28)
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);

        // Should be within bounds
        assert!(h * w >= processor.min_pixels());
        assert!(h * w <= processor.max_pixels());
    }

    #[test]
    fn test_smart_resize_too_large() {
        let processor = Qwen2VLProcessor::new();

        // Very large image
        let (h, w) = processor.smart_resize(3000, 3000).unwrap();

        // Should be scaled down
        assert!(h * w <= processor.max_pixels());
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_smart_resize_too_small() {
        let processor = Qwen2VLProcessor::new();

        // Small image (but above minimum dimension)
        let (h, w) = processor.smart_resize(100, 100).unwrap();

        // Should be scaled up to min_pixels
        assert!(h * w >= processor.min_pixels());
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_smart_resize_aspect_ratio_preserved() {
        let processor = Qwen2VLProcessor::new();

        // 2:1 aspect ratio
        let (h, w) = processor.smart_resize(400, 800).unwrap();

        // Aspect ratio should be approximately preserved
        let original_ratio = 800.0 / 400.0;
        let new_ratio = w as f64 / h as f64;
        assert!((new_ratio - original_ratio).abs() < 0.5);
    }

    #[test]
    fn test_smart_resize_extreme_aspect_ratio_error() {
        let processor = Qwen2VLProcessor::new();

        // 300:1 aspect ratio - should fail
        let result = processor.smart_resize(100, 30000);
        assert!(result.is_err());
    }

    #[test]
    fn test_smart_resize_small_dimension_clamps_to_factor() {
        let processor = Qwen2VLProcessor::new();

        // Dimension smaller than factor (28) should be clamped up, not rejected
        let (h, w) = processor.smart_resize(10, 100).unwrap();
        assert!(h >= 28);
        assert!(w >= 28);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_calculate_grid_thw_image() {
        let processor = Qwen2VLProcessor::new();

        // 448x448 image (16x16 grid patches)
        let (t, h, w) = processor.calculate_grid_thw(448, 448, 1);

        assert_eq!(t, 1); // Single image
        assert_eq!(h, 448 / 14); // 32
        assert_eq!(w, 448 / 14); // 32
    }

    #[test]
    fn test_calculate_tokens() {
        let processor = Qwen2VLProcessor::new();

        // With merge_size=2, tokens = (t * h * w) / 4
        let tokens = processor.calculate_tokens_from_grid(1, 32, 32);
        assert_eq!(tokens, (32 * 32) / 4); // 256
    }

    #[test]
    fn test_qwen2_vl_preprocess() {
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_normalize: Some(true),
            image_mean: Some(CLIP_MEAN.to_vec()),
            image_std: Some(CLIP_STD.to_vec()),
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            min_pixels: Some(DEFAULT_MIN_PIXELS),
            max_pixels: Some(DEFAULT_MAX_PIXELS),
            ..Default::default()
        };
        let processor = Qwen2VLProcessor::from_preprocessor_config(&config);

        let image = create_test_image(600, 400, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[&image]).unwrap();

        // encoder_input is patchified: [total_patches, patch_features]
        assert_eq!(result.encoder_input.ndim(), 2);
        assert!(result.encoder_input.shape()[0] > 0); // total_patches > 0

        // Check pixel values are normalized
        let flat = result.encoder_input_flat();
        // After normalization with CLIP mean/std, gray (0.5) should be near 0
        // (0.5 - 0.48) / 0.27 ≈ 0.07
        assert!(flat.iter().all(|&v| v.abs() < 1.0)); // Should be normalized

        // Check image_grid_thw and patches_per_image are present
        assert!(result.model_specific.contains_key("image_grid_thw"));
        assert!(result.model_specific.contains_key("patches_per_image"));

        // Verify token count is reasonable
        assert!(result.feature_token_counts[0] > 0);
    }

    /// `do_rescale=false, do_normalize=false` is the raw-pixel contract of an
    /// engine that rescales and normalizes on device: the patches carry the
    /// image's own 0..255 values, exactly.
    #[test]
    fn test_qwen2_vl_preprocess_raw_pixels() {
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_rescale: Some(false),
            do_normalize: Some(false),
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            min_pixels: Some(DEFAULT_MIN_PIXELS),
            max_pixels: Some(DEFAULT_MAX_PIXELS),
            ..Default::default()
        };
        let processor = Qwen2VLProcessor::from_preprocessor_config(&config);
        let image = create_test_image(600, 400, Rgb([17, 128, 255]));
        let result = processor.preprocess(&[&image]).unwrap();
        let flat = result.encoder_input_flat();
        assert!(!flat.is_empty());
        assert!(flat
            .iter()
            .all(|&v| v == v.round() && (0.0..=255.0).contains(&v)));
        // Every channel value of a solid image survives untouched.
        let mut seen: Vec<i32> = flat.iter().map(|&v| v as i32).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen, vec![17, 128, 255]);
    }

    #[test]
    fn test_qwen2_vl_preprocess_multiple() {
        let processor = Qwen2VLProcessor::new();
        let images = [
            create_test_image(600, 400, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = processor
            .preprocess(&images.iter().collect::<Vec<_>>())
            .unwrap();

        assert_two_image_batch(&result);
    }

    #[test]
    fn test_qwen2_vl_from_config() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(16),
                width: Some(16),
            }),
            merge_size: Some(4),
            min_pixels: Some(100000),
            max_pixels: Some(500000),
            temporal_patch_size: Some(4),
            ..Default::default()
        };

        let processor = Qwen2VLProcessor::from_preprocessor_config(&config);

        assert_eq!(processor.patch_size(), 16);
        assert_eq!(processor.merge_size(), 4);
        assert_eq!(processor.min_pixels(), 100000);
        assert_eq!(processor.max_pixels(), 500000);
        assert_eq!(processor.temporal_patch_size(), 4);
    }

    #[test]
    fn token_count_refuses_the_aspect_ratio_the_resize_refuses() {
        // An aspect ratio past the reference's limit fails the resize and
        // therefore the preprocessing; the count fails the same way instead
        // of counting the image at the minimum size.
        let processor = Qwen2VLProcessor::new();
        let error = processor
            .calculate_num_tokens(4000, 10)
            .unwrap_err()
            .to_string();
        assert!(error.contains("aspect ratio"), "{error}");
    }

    #[test]
    fn test_calculate_num_tokens_honors_config_max_pixels() {
        let processor = Qwen2VLProcessor::new();

        // A 1400x1400 image clamps to max_pixels before the grid is computed,
        // so a lower config max_pixels must yield fewer tokens.
        let default_tokens = processor.calculate_num_tokens(1400, 1400).unwrap();
        assert_eq!(default_tokens, 1225); // resized to 980x980 -> (70*70)/4

        let config = PreProcessorConfig {
            max_pixels: Some(512 * 28 * 28), // 401,408, below the 1,003,520 default
            ..Default::default()
        };
        let config_tokens = Qwen2VLProcessor::from_preprocessor_config(&config)
            .calculate_num_tokens(1400, 1400)
            .unwrap();
        assert_eq!(config_tokens, 484); // resized to 616x616 -> (44*44)/4
        assert_ne!(
            config_tokens, default_tokens,
            "config max_pixels override must change the token count"
        );
    }

    #[test]
    fn test_preprocess_honors_config_max_pixels() {
        let image = create_test_image(1400, 1400, Rgb([128, 128, 128]));

        let config = PreProcessorConfig {
            max_pixels: Some(512 * 28 * 28),
            ..Default::default()
        };
        let processor = Qwen2VLProcessor::from_preprocessor_config(&config);
        let result = processor.preprocess(&[&image]).unwrap();

        // 616x616 -> grid (1, 44, 44) -> (44*44)/4 = 484 tokens, vs 1225 at the default.
        assert_eq!(result.feature_token_counts[0], 484);
        if let Some(ModelSpecificValue::IntTensor { data, .. }) =
            result.model_specific.get("image_grid_thw")
        {
            assert_eq!(data, &[1, 44, 44]);
        } else {
            panic!("Expected image_grid_thw to be IntTensor");
        }
    }

    #[test]
    fn test_model_name() {
        let processor = Qwen2VLProcessor::new();
        assert_eq!(processor.model_name(), "qwen2-vl");
    }

    #[test]
    fn test_default_mean_std() {
        let processor = Qwen2VLProcessor::new();
        assert_eq!(processor.default_mean(), CLIP_MEAN);
        assert_eq!(processor.default_std(), CLIP_STD);
    }
}
