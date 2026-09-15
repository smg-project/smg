//! DeepSeek V4 Vision preprocessing.
//!
//! Pixel values are rounded to BF16 here and carried as exactly representable
//! F32 values through the vendor-neutral preprocessing interface. TokenSpeed's
//! default BF16 wire serialization therefore reproduces the reference tensor
//! bit-for-bit without placing a vendor dtype in the runtime boundary.

use std::collections::HashMap;

use image::{DynamicImage, GenericImageView, Rgb};
use ndarray::{ArrayD, IxDyn};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::VisionPreProcessor,
        transforms::{pad_contain_pil, resize_bicubic_pil, TransformError},
    },
};

const PATCH_SIZE: usize = 14;
const DOWNSAMPLE_RATIO: usize = 3;
const MAX_N_TOKEN: usize = 384;
const MIN_PIXELS: usize = 147_456;
const MAX_WH_RATIO: usize = 8;
const COMPRESS_PAD_TO: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepseekV4Geometry {
    pub best_height: usize,
    pub best_width: usize,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    pub num_tokens: usize,
    pub initial_num_tokens: usize,
    pub safe_resize_iterations: usize,
    pub aspect_clamped: bool,
    pub min_pixel_upscale: bool,
}

#[derive(Debug, Clone, Copy)]
struct Params {
    patch: usize,
    downsample: usize,
    max_tokens: usize,
    min_pixels: usize,
    max_wh_ratio: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            patch: PATCH_SIZE,
            downsample: DOWNSAMPLE_RATIO,
            max_tokens: MAX_N_TOKEN,
            min_pixels: MIN_PIXELS,
            max_wh_ratio: MAX_WH_RATIO,
        }
    }
}

fn grid_tokens(
    best_height: usize,
    best_width: usize,
    patch: usize,
    downsample: usize,
) -> (usize, usize, usize) {
    let n_llm_h = (best_height / patch).div_ceil(downsample);
    let n_llm_w = (best_width / patch).div_ceil(downsample);
    let mut num_tokens = n_llm_h * (n_llm_w + 1) + 2;
    if n_llm_h % 2 == 1 {
        num_tokens += n_llm_w + 1;
    }
    num_tokens += n_llm_h.div_ceil(2) * (n_llm_w + 1) % 2 * 2;
    (n_llm_h, n_llm_w, num_tokens)
}

fn solve_resize_ratio(
    height: usize,
    width: usize,
    params: Params,
    budget: usize,
) -> (usize, usize, usize, usize, usize) {
    let ratio = height as f64 / width as f64;
    let max_w_float = (((budget - 2) as f64 / ratio) + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * ratio;
    let (best_width, best_height) = if max_w_float < 1.0 {
        let max_w = 1;
        let mut max_h = (budget - 2) / (max_w + 1);
        if max_h % 2 == 1 {
            max_h -= 1;
        }
        (
            max_w * params.patch * params.downsample,
            max_h * params.patch * params.downsample,
        )
    } else if max_h_float < 2.0 {
        let max_h = 2;
        let max_w = (budget - 2) / max_h - 1;
        (
            max_w * params.patch * params.downsample,
            max_h * params.patch * params.downsample,
        )
    } else {
        let max_w = max_w_float.floor() as usize;
        let mut max_h = max_h_float.floor() as usize;
        if max_h % 2 == 1 {
            max_h -= 1;
        }
        let beta = (max_w * params.patch * params.downsample) as f64 / width as f64;
        let beta = beta.min((max_h * params.patch * params.downsample) as f64 / height as f64);
        (
            ((width as f64 * beta / params.patch as f64).floor() as usize) * params.patch,
            ((height as f64 * beta / params.patch as f64).floor() as usize) * params.patch,
        )
    };
    let (n_llm_h, n_llm_w, num_tokens) =
        grid_tokens(best_height, best_width, params.patch, params.downsample);
    (n_llm_h, n_llm_w, best_height, best_width, num_tokens)
}

fn geometry(width: usize, height: usize, params: Params) -> DeepseekV4Geometry {
    let mut logical_width = width;
    let mut logical_height = height;
    let aspect_clamped = logical_width > logical_height * params.max_wh_ratio;
    if aspect_clamped {
        logical_width = logical_height * params.max_wh_ratio;
    }
    let min_pixel_upscale =
        logical_width * logical_height > 0 && logical_width * logical_height < params.min_pixels;
    if min_pixel_upscale {
        let ratio = (params.min_pixels as f64 / (logical_width * logical_height) as f64).sqrt();
        logical_width = (logical_width as f64 * ratio) as usize;
        logical_height = (logical_height as f64 * ratio) as usize;
    }
    let mut best_width = logical_width.div_ceil(params.patch) * params.patch;
    let mut best_height = logical_height.div_ceil(params.patch) * params.patch;
    let (mut n_llm_h, mut n_llm_w, initial_num_tokens) =
        grid_tokens(best_height, best_width, params.patch, params.downsample);
    let max_tokens = params.max_tokens - (COMPRESS_PAD_TO - 1);
    let mut num_tokens = initial_num_tokens;
    let mut budget = max_tokens;
    let mut iterations = 0;
    while num_tokens > max_tokens {
        (n_llm_h, n_llm_w, best_height, best_width, num_tokens) =
            solve_resize_ratio(logical_height, logical_width, params, budget);
        budget -= 1;
        iterations += 1;
    }
    DeepseekV4Geometry {
        best_height,
        best_width,
        n_vit_h: best_height / params.patch,
        n_vit_w: best_width / params.patch,
        n_llm_h,
        n_llm_w,
        num_tokens,
        initial_num_tokens,
        safe_resize_iterations: iterations,
        aspect_clamped,
        min_pixel_upscale,
    }
}

/// Exposed for geometry and block-length contract tests.
pub fn deepseek_v4_geometry(width: u32, height: u32) -> DeepseekV4Geometry {
    geometry(width as usize, height as usize, Params::default())
}

#[inline]
fn round_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let least_significant_kept_bit = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + least_significant_kept_bit) & 0xffff_0000)
}

fn patchify(image: &DynamicImage, geometry: DeepseekV4Geometry, patch: usize) -> Vec<f32> {
    let rgb = image.to_rgb8();
    let mut patches = Vec::with_capacity(geometry.n_vit_h * geometry.n_vit_w * 3 * patch * patch);
    for patch_row in 0..geometry.n_vit_h {
        for patch_col in 0..geometry.n_vit_w {
            for channel in 0..3 {
                for row in 0..patch {
                    for col in 0..patch {
                        let pixel = rgb.get_pixel(
                            (patch_col * patch + col) as u32,
                            (patch_row * patch + row) as u32,
                        );
                        let value = (f32::from(pixel[channel]) / 255.0 - 0.5) / 0.5;
                        patches.push(round_to_bf16(value));
                    }
                }
            }
        }
    }
    patches
}

#[derive(Debug, Default)]
pub struct DeepseekV4VisionProcessor;

impl DeepseekV4VisionProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl VisionPreProcessor for DeepseekV4VisionProcessor {
    fn default_mean(&self) -> [f64; 3] {
        [0.5; 3]
    }

    fn default_std(&self) -> [f64; 3] {
        [0.5; 3]
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        _config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let params = Params::default();
        let mut all_patches = Vec::new();
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut patches_per_image = Vec::with_capacity(images.len());
        let mut n_vit_h = Vec::with_capacity(images.len());
        let mut n_vit_w = Vec::with_capacity(images.len());
        let mut n_llm_h = Vec::with_capacity(images.len());
        let mut n_llm_w = Vec::with_capacity(images.len());
        let mut resize_iterations = Vec::with_capacity(images.len());

        for image in images {
            let (width, height) = image.dimensions();
            let geo = geometry(width as usize, height as usize, params);
            let transformed = if width as usize >= params.max_wh_ratio * height as usize {
                resize_bicubic_pil(image, geo.best_width as u32, geo.best_height as u32)
            } else {
                pad_contain_pil(
                    image,
                    geo.best_width as u32,
                    geo.best_height as u32,
                    Rgb([127, 127, 127]),
                )
            };
            all_patches.extend(patchify(&transformed, geo, params.patch));
            item_sizes.push((height, width));
            feature_token_counts.push(geo.num_tokens);
            patches_per_image.push((geo.n_vit_h * geo.n_vit_w) as u32);
            n_vit_h.push(geo.n_vit_h as u32);
            n_vit_w.push(geo.n_vit_w as u32);
            n_llm_h.push(geo.n_llm_h as u32);
            n_llm_w.push(geo.n_llm_w as u32);
            resize_iterations.push(geo.safe_resize_iterations as u32);
        }
        let patch_count = patches_per_image.iter().map(|&n| n as usize).sum();
        let encoder_input = ArrayD::from_shape_vec(
            IxDyn(&[patch_count, 3, params.patch, params.patch]),
            all_patches,
        )
        .map_err(|error| TransformError::ShapeError(error.to_string()))?;
        let model_specific = HashMap::from([
            (
                "patches_per_image".to_string(),
                ModelSpecificValue::uint_1d(patches_per_image),
            ),
            ("n_vit_h".to_string(), ModelSpecificValue::uint_1d(n_vit_h)),
            ("n_vit_w".to_string(), ModelSpecificValue::uint_1d(n_vit_w)),
            ("n_llm_h".to_string(), ModelSpecificValue::uint_1d(n_llm_h)),
            ("n_llm_w".to_string(), ModelSpecificValue::uint_1d(n_llm_w)),
            (
                "safe_resize_iterations".to_string(),
                ModelSpecificValue::uint_1d(resize_iterations),
            ),
        ]);
        Ok(PreprocessedEncoderInputs {
            encoder_input,
            feature_token_counts,
            item_sizes,
            model_specific,
        })
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &PreProcessorConfig) -> usize {
        deepseek_v4_geometry(width, height).num_tokens
    }

    fn model_name(&self) -> &'static str {
        "deepseek_v4_vision"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_fixture_geometries_match_reference() {
        let cases = [
            ((64, 64), (28, 28, 10, 10, 114, 0)),
            ((1024, 768), (47, 63, 16, 21, 354, 1)),
            ((1024, 701), (42, 61, 14, 21, 310, 1)),
            ((450, 308), (23, 34, 8, 12, 106, 0)),
        ];
        for ((width, height), expected) in cases {
            let geo = deepseek_v4_geometry(width, height);
            assert_eq!(
                (
                    geo.n_vit_h,
                    geo.n_vit_w,
                    geo.n_llm_h,
                    geo.n_llm_w,
                    geo.num_tokens,
                    geo.safe_resize_iterations,
                ),
                expected,
                "{width}x{height}"
            );
        }
    }
}
