//! GLM-5.3-Flash image and video preprocessing.
//!
//! GLM owns its aspect-preserving resize and padding policy. Only generic
//! image transforms are shared; no Qwen processor types or policies leak in.

use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};
use ndarray::{Array2, Array3, IxDyn};

use crate::{
    encoder_inputs::ModelSpecificValue,
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::{
            pil_to_filter, resize, resize_bicubic_pil, stack_batch, to_tensor_and_normalize,
            TransformError,
        },
    },
};

const PATCH_SIZE: usize = 14;
const MERGE_SIZE: usize = 2;
const TEMPORAL_PATCH_SIZE: usize = 2;
const MIN_IMAGE_TOKENS: usize = 16;
const MAX_IMAGE_TOKENS: usize = 8000;
const MAX_VIDEO_TOKENS: usize = 240_000;

#[derive(Clone, Copy)]
struct Params {
    patch: usize,
    merge: usize,
    temporal: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
    mean: [f64; 3],
    std: [f64; 3],
}

#[derive(Clone, Copy)]
struct Geometry {
    target: (usize, usize),
    content: (usize, usize),
}

struct Patches {
    values: Vec<f32>,
    grid: (usize, usize, usize),
    count: usize,
    features: usize,
    tokens: usize,
}

impl Params {
    fn from_config(config: &PreProcessorConfig, video: bool) -> Self {
        let patch = config.get_patch_size(PATCH_SIZE);
        let merge = config.merge_size.unwrap_or(MERGE_SIZE);
        let temporal = config.temporal_patch_size.unwrap_or(TEMPORAL_PATCH_SIZE);
        let expand = config
            .get_extra::<usize>("patch_expand_factor")
            .unwrap_or(1);
        let pixels_per_token = temporal * (patch * merge).pow(2);
        let token_budget = |tokens| tokens * pixels_per_token;
        let min_pixels = config
            .min_pixels
            .or_else(|| {
                config
                    .get_extra::<usize>("min_image_tokens")
                    .map(token_budget)
            })
            .unwrap_or_else(|| token_budget(MIN_IMAGE_TOKENS));
        let max_pixels = config
            .max_pixels
            .or_else(|| {
                config
                    .get_extra::<usize>("max_image_tokens")
                    .map(token_budget)
            })
            .unwrap_or_else(|| {
                token_budget(if video {
                    MAX_VIDEO_TOKENS
                } else {
                    MAX_IMAGE_TOKENS
                })
            });
        let triple = |values: &Option<Vec<f64>>, default| {
            values
                .as_ref()
                .filter(|values| values.len() >= 3)
                .map(|values| [values[0], values[1], values[2]])
                .unwrap_or(default)
        };
        Self {
            patch,
            merge,
            temporal,
            factor: patch * merge * expand,
            min_pixels,
            max_pixels,
            mean: triple(&config.image_mean, PreProcessorConfig::CLIP_MEAN),
            std: triple(&config.image_std, PreProcessorConfig::CLIP_STD),
        }
    }

    fn geometry(
        self,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Geometry, TransformError> {
        if frames == 0
            || height == 0
            || width == 0
            || self.factor == 0
            || self.min_pixels == 0
            || self.min_pixels > self.max_pixels
        {
            return Err(TransformError::InvalidShape {
                expected: "valid GLM dimensions and pixel budget".to_string(),
                actual: vec![frames, height, width, self.min_pixels, self.max_pixels],
            });
        }
        let align_up = |value: usize| value.div_ceil(self.factor) * self.factor;
        let aligned_frames = (round_half_to_even(frames as f64 / self.temporal as f64) as usize
            * self.temporal)
            .max(self.temporal);
        let volume = |h: usize, w: usize| aligned_frames as u128 * h as u128 * w as u128;
        let mut target_h = align_up(height);
        let mut target_w = align_up(width);
        let source_volume = aligned_frames as f64 * height as f64 * width as f64;

        if volume(target_h, target_w) > self.max_pixels as u128 {
            let scale = (source_volume / self.max_pixels as f64).sqrt();
            target_h = ((height as f64 / scale / self.factor as f64).floor() as usize
                * self.factor)
                .max(self.factor);
            target_w = ((width as f64 / scale / self.factor as f64).floor() as usize * self.factor)
                .max(self.factor);
        } else if volume(target_h, target_w) < self.min_pixels as u128 {
            let scale = (self.min_pixels as f64 / source_volume).sqrt();
            target_h = align_up((height as f64 * scale).ceil() as usize);
            target_w = align_up((width as f64 * scale).ceil() as usize);
        }

        let mut scale = (target_h as f64 / height as f64).min(target_w as f64 / width as f64);
        if frames as u128 * height as u128 * width as u128 >= self.min_pixels as u128 {
            scale = scale.min(1.0);
        }
        Ok(Geometry {
            target: (target_h, target_w),
            content: (
                ((height as f64 * scale).floor() as usize).clamp(1, target_h),
                ((width as f64 * scale).floor() as usize).clamp(1, target_w),
            ),
        })
    }

    fn normalization(self, config: &PreProcessorConfig) -> ([f64; 3], [f64; 3]) {
        if config.do_normalize.unwrap_or(true) {
            (self.mean, self.std)
        } else {
            ([0.0; 3], [1.0; 3])
        }
    }
}

#[inline]
fn round_half_to_even(value: f64) -> f64 {
    let rounded = value.round();
    if (value.fract() - 0.5).abs() < 1e-9 && rounded as i64 % 2 != 0 {
        rounded - 1.0
    } else {
        rounded
    }
}

fn prepare(image: &DynamicImage, geometry: Geometry, filter: FilterType) -> DynamicImage {
    let (content_h, content_w) = geometry.content;
    let resized = if image.dimensions() == (content_w as u32, content_h as u32) {
        image.to_rgb8()
    } else if filter == FilterType::CatmullRom {
        resize_bicubic_pil(image, content_w as u32, content_h as u32).to_rgb8()
    } else {
        resize(image, content_w as u32, content_h as u32, filter).to_rgb8()
    };
    let (target_h, target_w) = geometry.target;
    if geometry.content == geometry.target {
        return DynamicImage::ImageRgb8(resized);
    }
    let mut canvas = RgbImage::new(target_w as u32, target_h as u32);
    image::imageops::overlay(&mut canvas, &resized, 0, 0);
    DynamicImage::ImageRgb8(canvas)
}

fn patchify(tensors: &[Array3<f32>], params: Params) -> Result<Patches, TransformError> {
    let Some(first) = tensors.first() else {
        return Err(TransformError::EmptyBatch);
    };
    let [channels, height, width] = first.shape() else {
        return Err(TransformError::InvalidShape {
            expected: "GLM tensor [3, H, W]".to_string(),
            actual: first.shape().to_vec(),
        });
    };
    if *channels != 3
        || height % params.factor != 0
        || width % params.factor != 0
        || tensors.iter().any(|tensor| tensor.shape() != first.shape())
    {
        return Err(TransformError::InvalidShape {
            expected: format!("aligned GLM RGB tensors with factor {}", params.factor),
            actual: first.shape().to_vec(),
        });
    }
    let grid_t = tensors.len().div_ceil(params.temporal);
    let grid_h = height / params.patch;
    let grid_w = width / params.patch;
    let count = grid_t * grid_h * grid_w;
    let features = 3 * params.temporal * params.patch * params.patch;
    let mut padded = tensors.to_vec();
    let last = tensors[tensors.len() - 1].clone();
    padded.resize_with(grid_t * params.temporal, || last.clone());
    let shaped = stack_batch(&padded)?
        .into_shape_with_order(IxDyn(&[
            grid_t,
            params.temporal,
            3,
            grid_h / params.merge,
            params.merge,
            params.patch,
            grid_w / params.merge,
            params.merge,
            params.patch,
        ]))
        .map_err(|error| TransformError::ShapeError(format!("GLM patch reshape: {error}")))?;
    let values = shaped
        .permuted_axes(IxDyn(&[0, 3, 6, 4, 7, 2, 1, 5, 8]))
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset()
        .0;
    Ok(Patches {
        values,
        grid: (grid_t, grid_h, grid_w),
        count,
        features,
        tokens: count / params.merge.pow(2),
    })
}

fn encode(
    frames: &[DynamicImage],
    budget_frames: usize,
    params: Params,
    config: &PreProcessorConfig,
) -> Result<Patches, TransformError> {
    let Some(first) = frames.first() else {
        return Err(TransformError::EmptyBatch);
    };
    let size = first.dimensions();
    if frames.iter().any(|frame| frame.dimensions() != size) {
        return Err(TransformError::ShapeError(
            "GLM video frames must have identical dimensions".to_string(),
        ));
    }
    let geometry = if config.do_resize.unwrap_or(true) {
        params.geometry(budget_frames, size.1 as usize, size.0 as usize)?
    } else {
        Geometry {
            target: (size.1 as usize, size.0 as usize),
            content: (size.1 as usize, size.0 as usize),
        }
    };
    let filter = pil_to_filter(config.resampling.or(Some(3)));
    let (mean, std) = params.normalization(config);
    let tensors = frames
        .iter()
        .map(|frame| to_tensor_and_normalize(&prepare(frame, geometry, filter), &mean, &std))
        .collect::<Vec<_>>();
    patchify(&tensors, params)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Glm53FlashProcessor;

impl Glm53FlashProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl VisionPreProcessor for Glm53FlashProcessor {
    fn default_mean(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let params = Params::from_config(config, false);
        let mut values = Vec::new();
        let mut grids = Vec::with_capacity(images.len() * 3);
        let mut counts = Vec::with_capacity(images.len());
        let mut tokens = Vec::with_capacity(images.len());
        let mut total = 0;
        for image in images {
            let patches = encode(std::slice::from_ref(image), params.temporal, params, config)?;
            values.extend(patches.values);
            grids.extend([
                patches.grid.0 as i64,
                patches.grid.1 as i64,
                patches.grid.2 as i64,
            ]);
            counts.push(patches.count as i64);
            tokens.push(patches.tokens);
            total += patches.count;
        }
        let features = 3 * params.temporal * params.patch.pow(2);
        let tensor = Array2::from_shape_vec((total, features), values).map_err(|error| {
            TransformError::ShapeError(format!("failed to build GLM image patches: {error}"))
        })?;
        Ok(PreprocessedEncoderInputs::new(
            tensor,
            tokens,
            images.iter().map(DynamicImage::dimensions).collect(),
        )
        .with_extra(
            "image_grid_thw",
            ModelSpecificValue::int_2d(grids, images.len(), 3),
        )
        .with_extra("patches_per_image", ModelSpecificValue::int_1d(counts)))
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let params = Params::from_config(config, true);
        let original = frames
            .first()
            .ok_or(TransformError::EmptyBatch)?
            .dimensions();
        let patches = encode(frames, frames.len(), params, config)?;
        let fps = config.get_extra::<f32>("fps").unwrap_or(2.0);
        if !fps.is_finite() || fps <= 0.0 {
            return Err(TransformError::ShapeError(format!(
                "GLM video fps must be positive, got {fps}"
            )));
        }
        let (grid_t, grid_h, grid_w) = patches.grid;
        let tensor = Array2::from_shape_vec((patches.count, patches.features), patches.values)
            .map_err(|error| {
                TransformError::ShapeError(format!("failed to build GLM video patches: {error}"))
            })?;
        Ok(
            PreprocessedEncoderInputs::new(tensor, vec![patches.tokens], vec![original])
                .with_extra(
                    "video_grid_thw",
                    ModelSpecificValue::int_2d(
                        vec![grid_t as i64, grid_h as i64, grid_w as i64],
                        1,
                        3,
                    ),
                )
                .with_extra(
                    "patches_per_video",
                    ModelSpecificValue::int_1d(vec![patches.count as i64]),
                )
                .with_extra(
                    "patches_per_image",
                    ModelSpecificValue::int_1d(vec![patches.count as i64]),
                )
                .with_extra(
                    "video_second_per_grid",
                    ModelSpecificValue::Tensor {
                        data: vec![params.temporal as f32 / fps],
                        shape: vec![1],
                    },
                ),
        )
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let images = frames
            .iter()
            .map(|frame| {
                RgbImage::from_raw(frame.width, frame.height, frame.data.to_vec())
                    .map(DynamicImage::ImageRgb8)
                    .ok_or_else(|| {
                        TransformError::ShapeError(format!(
                            "invalid GLM RGB frame {}x{} with {} bytes",
                            frame.width,
                            frame.height,
                            frame.data.len()
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_video(&images, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let params = Params::from_config(config, false);
        params
            .geometry(params.temporal, height as usize, width as usize)
            .map(|geometry| {
                geometry.target.0 / params.patch * (geometry.target.1 / params.patch)
                    / params.merge.pow(2)
            })
            .unwrap_or(0)
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
            .preprocess(&[image], &config(MAX_IMAGE_TOKENS))
            .unwrap();

        assert_eq!(grid(&output, "image_grid_thw"), vec![1, 16, 30]);
        assert_eq!(output.feature_token_counts, vec![120]);
        assert_eq!(output.encoder_input.shape(), &[480, 1176]);
        let red =
            ((1.0 - PreProcessorConfig::CLIP_MEAN[0]) / PreProcessorConfig::CLIP_STD[0]) as f32;
        let padding = (-PreProcessorConfig::CLIP_MEAN[0] / PreProcessorConfig::CLIP_STD[0]) as f32;
        assert!(output.encoder_input.iter().any(|v| (*v - red).abs() < 1e-5));
        assert!(output
            .encoder_input
            .iter()
            .any(|v| (*v - padding).abs() < 1e-5));
    }

    #[test]
    fn small_images_expand_to_the_minimum_token_budget() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([1, 2, 3])));
        let output = Glm53FlashProcessor::new()
            .preprocess(&[image], &config(MAX_IMAGE_TOKENS))
            .unwrap();

        assert_eq!(grid(&output, "image_grid_thw"), vec![1, 8, 8]);
        assert_eq!(output.feature_token_counts, vec![16]);
    }

    #[test]
    fn video_uses_temporal_patches_and_reports_timing() {
        let frame = DynamicImage::ImageRgb8(RgbImage::from_pixel(400, 200, Rgb([1, 2, 3])));
        let output = Glm53FlashProcessor::new()
            .preprocess_video(&vec![frame; 5], &config(MAX_VIDEO_TOKENS))
            .unwrap();

        assert_eq!(grid(&output, "video_grid_thw"), vec![3, 16, 30]);
        assert_eq!(output.feature_token_counts, vec![360]);
        match output.model_specific.get("video_second_per_grid").unwrap() {
            ModelSpecificValue::Tensor { data, .. } => assert_eq!(data, &[1.0]),
            value => panic!("unexpected timing value: {value:?}"),
        }
    }
}
