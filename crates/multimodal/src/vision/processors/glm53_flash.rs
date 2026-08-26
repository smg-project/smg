//! GLM-5.3-Flash image and video preprocessing.

use image::DynamicImage;

use super::qwen_vl_base::{
    QwenSpatialResizeMode, QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode,
};
use crate::{
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::TransformError,
    },
};

const DEFAULT_PATCH_SIZE: usize = 14;
const DEFAULT_MERGE_SIZE: usize = 2;
const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;
const DEFAULT_PATCH_EXPAND_FACTOR: usize = 1;
const DEFAULT_MIN_IMAGE_TOKENS: usize = 16;
const DEFAULT_MAX_IMAGE_TOKENS: usize = 8000;
const DEFAULT_MAX_VIDEO_TOKENS: usize = 240_000;

#[derive(Debug, Clone)]
pub struct Glm53FlashProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for Glm53FlashProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Glm53FlashProcessor {
    pub fn new() -> Self {
        Self::with_config(&PreProcessorConfig::default(), false)
    }

    fn with_config(config: &PreProcessorConfig, video: bool) -> Self {
        let patch_size = config.get_patch_size(DEFAULT_PATCH_SIZE);
        let merge_size = config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE);
        let temporal_patch_size = config
            .temporal_patch_size
            .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE);
        let patch_expand_factor = config
            .patch_expand_factor
            .unwrap_or(DEFAULT_PATCH_EXPAND_FACTOR);
        let pixels_per_token = temporal_patch_size * (patch_size * merge_size).pow(2);
        let min_pixels = config
            .min_pixels
            .or_else(|| {
                config
                    .min_image_tokens
                    .map(|tokens| tokens * pixels_per_token)
            })
            .unwrap_or(DEFAULT_MIN_IMAGE_TOKENS * pixels_per_token);
        let max_tokens = if video {
            DEFAULT_MAX_VIDEO_TOKENS
        } else {
            DEFAULT_MAX_IMAGE_TOKENS
        };
        let max_pixels = config
            .max_pixels
            .or_else(|| {
                config
                    .max_image_tokens
                    .map(|tokens| tokens * pixels_per_token)
            })
            .unwrap_or(max_tokens * pixels_per_token);

        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels,
                max_pixels,
                video_min_pixels: min_pixels,
                video_max_pixels: max_pixels,
                video_resize_mode: QwenVideoResizeMode::TotalVolume,
                spatial_resize_mode: QwenSpatialResizeMode::AlignedCanvas {
                    patch_expand_factor,
                },
                temporal_patch_size,
                mean: PreProcessorConfig::CLIP_MEAN,
                std: PreProcessorConfig::CLIP_STD,
                model_name: "glm-5.3-flash",
            }),
        }
    }
}

impl VisionPreProcessor for Glm53FlashProcessor {
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
        Self::with_config(config, false)
            .inner
            .preprocess(images, config)
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Self::with_config(config, true)
            .inner
            .preprocess_video(frames, config)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Self::with_config(config, true)
            .inner
            .preprocess_video_rgb(frames, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        Self::with_config(config, false)
            .inner
            .calculate_num_tokens(width, height, config)
    }

    fn model_name(&self) -> &'static str {
        "glm-5.3-flash"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::encoder_inputs::ModelSpecificValue;

    fn config(max_tokens: usize) -> PreProcessorConfig {
        PreProcessorConfig::from_json(&format!(
            r#"{{
                "do_resize": true,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "patch_size": 14,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "patch_expand_factor": 1,
                "min_image_tokens": 16,
                "max_image_tokens": {max_tokens}
            }}"#
        ))
        .unwrap()
    }

    fn grid(output: &PreprocessedEncoderInputs, key: &str) -> Vec<i64> {
        match output.model_specific.get(key).unwrap() {
            ModelSpecificValue::IntTensor { data, .. } => data.clone(),
            value => panic!("unexpected grid value: {value:?}"),
        }
    }

    #[test]
    fn preserves_aspect_ratio_on_an_aligned_canvas() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(400, 200, Rgb([255, 0, 0])));
        let output = Glm53FlashProcessor::new()
            .preprocess(&[image], &config(DEFAULT_MAX_IMAGE_TOKENS))
            .unwrap();

        assert_eq!(grid(&output, "image_grid_thw"), vec![1, 16, 30]);
        assert_eq!(output.feature_token_counts, vec![120]);
        assert_eq!(output.encoder_input.shape(), &[480, 1176]);

        let red =
            ((1.0 - PreProcessorConfig::CLIP_MEAN[0]) / PreProcessorConfig::CLIP_STD[0]) as f32;
        let padded_red =
            (-PreProcessorConfig::CLIP_MEAN[0] / PreProcessorConfig::CLIP_STD[0]) as f32;
        assert!(output
            .encoder_input
            .iter()
            .any(|value| (*value - red).abs() < 1e-5));
        assert!(output
            .encoder_input
            .iter()
            .any(|value| (*value - padded_red).abs() < 1e-5));
    }

    #[test]
    fn small_images_expand_to_the_minimum_token_budget() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([1, 2, 3])));
        let output = Glm53FlashProcessor::new()
            .preprocess(&[image], &config(DEFAULT_MAX_IMAGE_TOKENS))
            .unwrap();

        assert_eq!(grid(&output, "image_grid_thw"), vec![1, 8, 8]);
        assert_eq!(output.feature_token_counts, vec![16]);
    }

    #[test]
    fn video_uses_temporal_patches_and_reports_timing() {
        let frame = DynamicImage::ImageRgb8(RgbImage::from_pixel(400, 200, Rgb([1, 2, 3])));
        let frames = vec![frame; 5];
        let output = Glm53FlashProcessor::new()
            .preprocess_video(&frames, &config(DEFAULT_MAX_VIDEO_TOKENS))
            .unwrap();

        assert_eq!(grid(&output, "video_grid_thw"), vec![3, 16, 30]);
        assert_eq!(output.feature_token_counts, vec![360]);
        match output.model_specific.get("video_second_per_grid").unwrap() {
            ModelSpecificValue::Tensor { data, .. } => assert_eq!(data, &[1.0]),
            value => panic!("unexpected timing value: {value:?}"),
        }
    }
}
