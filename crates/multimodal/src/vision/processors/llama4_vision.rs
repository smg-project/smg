//! LLaMA 4 Vision image processor.
//!
//! This module implements the LLaMA 4 Vision (Llama-4-Scout, Llama-4-Maverick) image preprocessing
//! pipeline with tile-based processing similar to other dynamic resolution models.
//!
//! # Key Features
//!
//! | Feature | Value |
//! |---------|-------|
//! | Tile size | 336x336 |
//! | Default max_patches | 16 |
//! | Normalization | [0.5, 0.5, 0.5] mean/std |
//! | Interpolation | Bilinear |
//! | Global tile | Added when num_tiles > 1 |
//!
//! # Processing Pipeline
//!
//! 1. **Find supported resolutions**: Calculate valid tile configurations
//! 2. **Get best fit**: Find optimal resolution without distortion
//! 3. **Resize**: Scale to target resolution maintaining aspect ratio
//! 4. **Pad**: Add black padding (0) to reach target dimensions
//! 5. **Normalize**: Apply [0.5, 0.5, 0.5] mean/std normalization
//! 6. **Tile**: Split into (num_tiles_h * num_tiles_w, 3, 336, 336) tiles
//! 7. **Global tile**: If multiple tiles, add global view at the end
//!
//! # Token Count
//!
//! For LLaMA 4, tokens = num_tiles * (tile_size / patch_size)²
//! where patch_size is typically 14, giving 576 tokens per tile.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array4;

use crate::{
    buffer_pool,
    encoder_inputs::{EncoderDtype, EncoderInput, PixelNorm},
    pixel_convert::WireWriter,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
        rgb_runs::RgbSplitter,
        scratch,
        transforms::{self, TransformError},
    },
};

/// Default normalization mean for LLaMA 4 Vision.
pub(crate) const LLAMA4_MEAN: [f64; 3] = [0.5, 0.5, 0.5];

/// Default normalization std for LLaMA 4 Vision.
pub(crate) const LLAMA4_STD: [f64; 3] = [0.5, 0.5, 0.5];

/// Default tile size for LLaMA 4 Vision.
pub(crate) const TILE_SIZE: u32 = 336;

/// Default maximum number of patches/tiles.
pub(crate) const DEFAULT_MAX_PATCHES: usize = 16;

/// Patch size used in vision encoder.
pub(crate) const PATCH_SIZE: usize = 14;

/// LLaMA 4 Vision image processor.
///
/// Implements tile-based processing with dynamic resolution selection.
#[derive(Debug, Clone)]
pub struct Llama4VisionProcessor {
    /// Tile size (both height and width).
    tile_size: u32,
    /// Maximum number of tiles/patches.
    max_patches: usize,
    /// Whether to resize to max canvas (upscale aggressively).
    resize_to_max_canvas: bool,
    /// Normalization mean.
    mean: [f64; 3],
    /// Normalization std.
    std: [f64; 3],
}

impl Default for Llama4VisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Llama4VisionProcessor {
    /// Create a new LLaMA 4 Vision processor with default settings.
    pub(crate) fn new() -> Self {
        Self {
            tile_size: TILE_SIZE,
            max_patches: DEFAULT_MAX_PATCHES,
            resize_to_max_canvas: false,
            mean: LLAMA4_MEAN,
            std: LLAMA4_STD,
        }
    }

    /// Create a processor from preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            tile_size: config
                .size
                .as_ref()
                .and_then(|s| s.get("height").copied())
                .unwrap_or(TILE_SIZE),
            max_patches: config.max_image_tiles.unwrap_or(DEFAULT_MAX_PATCHES),
            resize_to_max_canvas: false,
            mean: config
                .image_mean
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(LLAMA4_MEAN),
            std: config
                .image_std
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(LLAMA4_STD),
        }
    }

    /// Refuse a tile size of zero, which the canvas divides by, and a tile
    /// count of zero, which leaves no canvas to pick.
    pub(crate) fn validate(&self) -> Result<(), TransformError> {
        let zero = if self.tile_size == 0 {
            "size.height"
        } else if self.max_patches == 0 {
            "max_image_tiles"
        } else {
            return Ok(());
        };
        Err(TransformError::ShapeError(format!(
            "Llama 4 {zero} must be positive"
        )))
    }

    /// Get all factors of a number.
    fn get_factors(n: usize) -> HashSet<usize> {
        let mut factors = HashSet::new();
        for i in 1..=(n as f64).sqrt() as usize {
            if n.is_multiple_of(i) {
                factors.insert(i);
                factors.insert(n / i);
            }
        }
        factors
    }

    /// Find all supported resolutions for the given max_patches.
    ///
    /// Returns list of (height, width) in pixels.
    fn find_supported_resolutions(&self) -> Vec<(u32, u32)> {
        let mut resolutions = Vec::new();
        let tile = self.tile_size;

        // For each possible number of chunks from max_patches down to 1
        for chunk_size in (1..=self.max_patches).rev() {
            let factors = Self::get_factors(chunk_size);
            for &factor in &factors {
                let h_tiles = factor;
                let w_tiles = chunk_size / factor;
                resolutions.push((h_tiles as u32 * tile, w_tiles as u32 * tile));
            }
        }

        resolutions
    }

    /// Get the maximum resolution without distortion.
    ///
    /// Given an image size and target size, compute the largest size
    /// that fits within target while maintaining aspect ratio.
    fn get_max_res_without_distortion(
        image_size: (u32, u32),
        target_size: (u32, u32),
    ) -> (u32, u32) {
        let (orig_h, orig_w) = image_size;
        let (target_h, target_w) = target_size;

        let scale_w = target_w as f64 / orig_w as f64;
        let scale_h = target_h as f64 / orig_h as f64;

        if scale_w < scale_h {
            let new_w = target_w;
            let new_h = (orig_h as f64 * scale_w).floor() as u32;
            (new_h.min(target_h), new_w)
        } else {
            let new_h = target_h;
            let new_w = (orig_w as f64 * scale_h).floor() as u32;
            (new_h, new_w.min(target_w))
        }
    }

    /// Find the best fitting resolution from supported resolutions.
    ///
    /// Selects resolution that:
    /// - Minimizes upscaling if possible (unless resize_to_max_canvas)
    /// - Minimizes downscaling if no upscaling possible
    /// - Minimizes padding area when tied
    fn get_best_fit(&self, image_size: (u32, u32)) -> (u32, u32) {
        let resolutions = self.find_supported_resolutions();
        let (orig_h, orig_w) = image_size;

        // Calculate scaling factors for each resolution
        let scales_and_resolutions: Vec<(f64, (u32, u32))> = resolutions
            .iter()
            .map(|&(target_h, target_w)| {
                let scale_w = target_w as f64 / orig_w as f64;
                let scale_h = target_h as f64 / orig_h as f64;
                // Limiting scale is the minimum (the side that constrains)
                let scale = scale_w.min(scale_h);
                (scale, (target_h, target_w))
            })
            .collect();

        // Separate upscaling and downscaling options
        let upscaling: Vec<_> = scales_and_resolutions
            .iter()
            .filter(|(s, _)| *s >= 1.0)
            .copied()
            .collect();

        let selected_scale = if upscaling.is_empty() {
            // No upscaling possible, pick largest downscaling (minimum reduction)
            scales_and_resolutions
                .iter()
                .filter(|(s, _)| *s < 1.0)
                .map(|(s, _)| *s)
                .fold(f64::NEG_INFINITY, f64::max)
        } else if self.resize_to_max_canvas {
            // Pick largest upscaling
            upscaling
                .iter()
                .map(|(s, _)| *s)
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            // Pick smallest upscaling (minimum distortion)
            upscaling
                .iter()
                .map(|(s, _)| *s)
                .fold(f64::INFINITY, f64::min)
        };

        // Get all resolutions with the selected scale
        let candidates: Vec<_> = scales_and_resolutions
            .iter()
            .filter(|(s, _)| (*s - selected_scale).abs() < 1e-9)
            .map(|(_, res)| *res)
            .collect();

        // If multiple candidates, pick the one with minimum area (less padding)
        if candidates.len() > 1 {
            *candidates
                .iter()
                .min_by_key(|(h, w)| h * w)
                .unwrap_or(&candidates[0])
        } else {
            candidates[0]
        }
    }

    /// Where an image of `(width, height)` lands: the size it is resized to
    /// and its canvas in tiles.
    fn plan(&self, (orig_w, orig_h): (u32, u32)) -> Plan {
        let image_size = (orig_h, orig_w);
        let tile = self.tile_size;
        let (target_h, target_w) = self.get_best_fit(image_size);

        // Upscaling stops at one tile unless the canvas is to be filled.
        let resize_target = if self.resize_to_max_canvas {
            (target_h, target_w)
        } else {
            (
                target_h.min(orig_h.max(tile)),
                target_w.min(orig_w.max(tile)),
            )
        };
        let (new_h, new_w) = Self::get_max_res_without_distortion(image_size, resize_target);

        Plan {
            resized: (new_w.max(1), new_h.max(1)),
            grid: ((target_h / tile) as usize, (target_w / tile) as usize),
        }
    }

    /// The plan of every image of the batch, in order.
    fn plans(&self, images: &[&DynamicImage]) -> Vec<Plan> {
        images
            .iter()
            .map(|image| self.plan(image.dimensions()))
            .collect()
    }

    /// `(byte - 255 * mean) / (255 * std)` in `f32`, the fused order of
    /// [`transforms::to_tensor_and_normalize`], in the form a wire carries.
    fn norm(&self) -> PixelNorm {
        let tile = self.tile_size as usize;
        PixelNorm {
            scale: 1.0,
            offset: self.mean.map(|mean| mean as f32 * 255.0),
            divisor: self.std.map(|std| std as f32 * 255.0),
            channel_block: tile * tile,
        }
    }

    /// One image's tiles in output order, each channel of a tile handed to
    /// `run` once as bytes: the canvas tiles, then the global view behind
    /// more than one tile.
    fn write_image(&self, image: &DynamicImage, plan: &Plan, mut run: impl FnMut(usize, &[u8])) {
        let tile = self.tile_size as usize;
        let (width, height) = plan.resized;
        // The resizer's same-size pass is a copy of the bytes.
        let resized = if image.dimensions() == plan.resized {
            Cow::Borrowed(image)
        } else {
            Cow::Owned(transforms::resize(
                image,
                width,
                height,
                FilterType::Triangle,
            ))
        };
        let (width, height, rgb) = transforms::rgb_bytes(&resized);
        tile_blocks(&rgb, (width, height), plan.grid, tile, &mut run);
        if plan.has_global() {
            let global =
                transforms::resize(image, self.tile_size, self.tile_size, FilterType::Triangle);
            let (width, height, rgb) = transforms::rgb_bytes(&global);
            tile_blocks(&rgb, (width, height), (1, 1), tile, &mut run);
        }
    }

    /// The batch's outputs around its tiles, the same for both forms of the
    /// encoder input.
    fn outputs(
        &self,
        images: &[&DynamicImage],
        plans: &[Plan],
        encoder_input: EncoderInput,
    ) -> PreprocessedEncoderInputs {
        let aspect_ratios = plans
            .iter()
            .flat_map(|plan| [plan.grid.0 as i64, plan.grid.1 as i64])
            .collect();
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "aspect_ratios".to_string(),
            ModelSpecificValue::IntTensor {
                data: aspect_ratios,
                shape: vec![images.len(), 2],
            },
        );
        model_specific.insert(
            "patches_per_image".to_string(),
            ModelSpecificValue::int_1d(plans.iter().map(|plan| plan.tiles() as i64).collect()),
        );

        PreprocessedEncoderInputs {
            encoder_input,
            feature_token_counts: plans
                .iter()
                .map(|plan| self.calculate_num_tokens_for_aspect_ratio(plan.grid))
                .collect(),
            item_sizes: images.iter().map(|image| image.dimensions()).collect(),
            model_specific,
        }
    }

    /// Calculate number of image tokens for a given aspect ratio.
    pub(crate) fn calculate_num_tokens_for_aspect_ratio(
        &self,
        aspect_ratio: (usize, usize),
    ) -> usize {
        let (h_tiles, w_tiles) = aspect_ratio;
        let num_tiles = h_tiles * w_tiles;
        // Add 1 for global tile if num_tiles > 1
        let total_tiles = if num_tiles > 1 {
            num_tiles + 1
        } else {
            num_tiles
        };
        let tokens_per_tile = (self.tile_size as usize / PATCH_SIZE).pow(2);
        total_tiles * tokens_per_tile
    }
}

/// Where one image lands in the output.
#[derive(Debug, Clone, Copy)]
struct Plan {
    /// Width and height the image is resized to, inside its canvas.
    resized: (u32, u32),
    /// Rows and columns of tiles of the canvas.
    grid: (usize, usize),
}

impl Plan {
    /// Whether the global view follows the canvas: behind more than one tile.
    fn has_global(&self) -> bool {
        self.grid.0 * self.grid.1 > 1
    }

    /// Tiles the image writes: its canvas and the global view.
    fn tiles(&self) -> usize {
        self.grid.0 * self.grid.1 + usize::from(self.has_global())
    }
}

/// The tiles of a `rows` x `columns` canvas in output order, `rgb` at its
/// top left and black (0) beyond it: `run` takes a tile's channel and that
/// channel's bytes.
fn tile_blocks(
    rgb: &[u8],
    (width, height): (usize, usize),
    (rows, columns): (usize, usize),
    tile: usize,
    mut run: impl FnMut(usize, &[u8]),
) {
    let splitter = RgbSplitter::detect();
    let block = tile * tile;
    // A pooled buffer: a full tile covers it, a partial one clears it first.
    let mut bytes = buffer_pool::take(3 * block);
    for tile_y in 0..rows {
        for tile_x in 0..columns {
            let left = (tile_x * tile).min(width);
            let top = (tile_y * tile).min(height);
            let (inside, filled) = ((width - left).min(tile), (height - top).min(tile));
            if inside < tile || filled < tile {
                bytes.fill(0);
            }
            let (r, rest) = bytes.split_at_mut(block);
            let (g, b) = rest.split_at_mut(block);
            for y in 0..filled {
                let row = &rgb[((top + y) * width + left) * 3..][..inside * 3];
                let at = y * tile..y * tile + inside;
                splitter.split(row, &mut r[at.clone()], &mut g[at.clone()], &mut b[at]);
            }
            for (channel, of_channel) in bytes.chunks_exact(block).enumerate() {
                run(channel, of_channel);
            }
        }
    }
    buffer_pool::recycle(bytes);
}

/// The request's error for a patch output that cannot be reserved.
fn refused(reason: impl std::fmt::Display) -> TransformError {
    TransformError::ReserveRefused(format!("cannot reserve Llama 4 patch output: {reason}"))
}

/// The product of `factors`, the size of a patch output, or the request's
/// error when `usize` cannot count it.
fn output_len(factors: &[usize]) -> Result<usize, TransformError> {
    factors
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| refused("its size overflows usize"))
}

impl VisionPreProcessor for Llama4VisionProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.mean
    }

    fn default_std(&self) -> [f64; 3] {
        self.std
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        TransformError::if_empty(images)?;

        let plans = self.plans(images);
        let tile = self.tile_size as usize;
        let tiles: usize = plans.iter().map(Plan::tiles).sum();
        let table = self.norm().table_f32();
        // Every image's tiles land in the one buffer the engine reads:
        // [total_tiles, C, H, W], no batch dimension, no zero-padding.
        let mut data =
            scratch::try_take_f32_cap(output_len(&[tiles, 3, tile, tile])?).map_err(refused)?;
        for (image, plan) in images.iter().zip(&plans) {
            self.write_image(image, plan, |channel, bytes| {
                let table = &table[channel];
                data.extend(bytes.iter().map(|&byte| table[usize::from(byte)]));
            });
        }

        let encoder_input = Array4::from_shape_vec((tiles, 3, tile, tile), data)
            .map_err(|e| TransformError::ShapeError(format!("Failed to shape tiles: {e}")))?;
        Ok(self.outputs(images, &plans, encoder_input.into()))
    }

    /// The batch written once in `dtype`: the wire's tables are
    /// [`preprocess`](Self::preprocess)'s own table narrowed, so every float
    /// dtype is served.
    fn preprocess_wire(
        &self,
        images: &[&DynamicImage],
        dtype: EncoderDtype,
    ) -> Option<Result<PreprocessedEncoderInputs, TransformError>> {
        let norm = self.norm();
        let writer = WireWriter::new(&norm, dtype)?;
        if images.is_empty() {
            return Some(Err(TransformError::EmptyBatch));
        }

        let plans = self.plans(images);
        let tile = self.tile_size as usize;
        let tiles: usize = plans.iter().map(Plan::tiles).sum();
        // A pooled buffer: its bytes are a previous request's until the tiles
        // are written, and the tiles cover it.
        let reserved = output_len(&[tile, tile, writer.element_size()]).and_then(|block_bytes| {
            let len = output_len(&[tiles, 3, block_bytes])?;
            Ok((block_bytes, buffer_pool::try_take(len).map_err(refused)?))
        });
        let (block_bytes, mut wire) = match reserved {
            Ok(reserved) => reserved,
            Err(error) => return Some(Err(error)),
        };
        let mut written = 0;
        for (image, plan) in images.iter().zip(&plans) {
            self.write_image(image, plan, |channel, bytes| {
                writer.write_run(&mut wire[written..written + block_bytes], bytes, channel);
                written += block_bytes;
            });
        }
        debug_assert_eq!(written, wire.len(), "the tiles cover the wire");

        Some(
            EncoderInput::wire(wire, dtype, vec![tiles, 3, tile, tile], norm)
                .map(|encoder_input| self.outputs(images, &plans, encoder_input))
                .map_err(TransformError::ShapeError),
        )
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let image_size = (height, width);
        // target_size from get_best_fit determines the canvas and tile count
        let target_size = self.get_best_fit(image_size);

        let tile = self.tile_size as usize;
        let num_tiles_h = target_size.0 as usize / tile;
        let num_tiles_w = target_size.1 as usize / tile;

        Ok(self.calculate_num_tokens_for_aspect_ratio((num_tiles_h, num_tiles_w)))
    }

    fn model_name(&self) -> &'static str {
        "llama4-vision"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        // For LLaMA 4, the size depends on the input image
        None
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;
    use crate::test_support::create_test_image;

    #[test]
    fn test_llama4_vision_processor_default() {
        let processor = Llama4VisionProcessor::new();
        assert_eq!(processor.tile_size, TILE_SIZE);
        assert_eq!(processor.max_patches, DEFAULT_MAX_PATCHES);
        assert_eq!(processor.mean, LLAMA4_MEAN);
        assert_eq!(processor.std, LLAMA4_STD);
    }

    #[test]
    fn test_get_factors() {
        let factors = Llama4VisionProcessor::get_factors(12);
        assert!(factors.contains(&1));
        assert!(factors.contains(&2));
        assert!(factors.contains(&3));
        assert!(factors.contains(&4));
        assert!(factors.contains(&6));
        assert!(factors.contains(&12));
        assert_eq!(factors.len(), 6);
    }

    #[test]
    fn test_find_supported_resolutions() {
        let processor = Llama4VisionProcessor {
            max_patches: 4,
            ..Llama4VisionProcessor::new()
        };
        let resolutions = processor.find_supported_resolutions();

        // Should include 1x1, 1x2, 2x1, 1x3, 3x1, 2x2, 1x4, 4x1
        let expected: Vec<(u32, u32)> = vec![
            (336, 336),  // 1x1
            (336, 672),  // 1x2
            (672, 336),  // 2x1
            (336, 1008), // 1x3
            (1008, 336), // 3x1
            (672, 672),  // 2x2
            (336, 1344), // 1x4
            (1344, 336), // 4x1
        ];

        for exp in expected {
            assert!(
                resolutions.contains(&exp),
                "Expected resolution {exp:?} not found"
            );
        }
    }

    #[test]
    fn test_get_best_fit_square() {
        let processor = Llama4VisionProcessor::new();
        let best = processor.get_best_fit((500, 500));
        // Square image should get a square or near-square resolution
        assert!(best.0 == best.1 || (best.0 as i32 - best.1 as i32).abs() <= 336);
    }

    #[test]
    fn test_get_best_fit_wide() {
        let processor = Llama4VisionProcessor::new();
        let best = processor.get_best_fit((300, 900));
        // Wide image should get wider resolution
        assert!(best.1 >= best.0);
    }

    #[test]
    fn test_get_best_fit_tall() {
        let processor = Llama4VisionProcessor::new();
        let best = processor.get_best_fit((900, 300));
        // Tall image should get taller resolution
        assert!(best.0 >= best.1);
    }

    #[test]
    fn test_preprocess_square_image() {
        let processor = Llama4VisionProcessor::new();

        let image = create_test_image(500, 500, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[&image]).unwrap();

        // 4D output: [total_tiles, C, H, W]
        assert_eq!(result.encoder_input.ndim(), 4);
        assert_eq!(result.feature_token_counts.len(), 1);
        assert!(result.feature_token_counts[0] > 0);

        // Check pixel values are normalized
        let flat = result.encoder_input_flat();
        assert!(flat.iter().all(|&v| (-1.5..=1.5).contains(&v)));
    }

    #[test]
    fn test_preprocess_wide_image() {
        let processor = Llama4VisionProcessor::new();

        let image = create_test_image(1000, 300, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[&image]).unwrap();

        // 4D output: [total_tiles, C, H, W]
        assert_eq!(result.encoder_input.ndim(), 4);
        assert_eq!(result.feature_token_counts.len(), 1);
        // Wide image should have more tiles in width direction
        let aspect_ratios = result.model_specific.get("aspect_ratios").unwrap();
        if let ModelSpecificValue::IntTensor { data, .. } = aspect_ratios {
            let h_tiles = data[0];
            let w_tiles = data[1];
            assert!(w_tiles >= h_tiles);
        }
    }

    #[test]
    fn test_preprocess_multiple_images() {
        let processor = Llama4VisionProcessor::new();

        let images = [
            create_test_image(500, 500, Rgb([100, 100, 100])),
            create_test_image(800, 400, Rgb([150, 150, 150])),
        ];

        let result = processor
            .preprocess(&images.iter().collect::<Vec<_>>())
            .unwrap();

        // 4D output: [total_tiles, C, H, W] — tiles from both images concatenated
        assert_eq!(result.encoder_input.ndim(), 4);
        assert_eq!(result.feature_token_counts.len(), 2);
        assert_eq!(result.item_sizes.len(), 2);
        // Total tiles should be > 2 (at least 1 tile per image)
        assert!(result.encoder_input.shape()[0] >= 2);
    }

    #[test]
    fn test_global_tile_added_for_multiple_tiles() {
        let processor = Llama4VisionProcessor::new();

        // Large image that will require multiple tiles
        let image = create_test_image(1000, 1000, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[&image]).unwrap();

        let aspect_ratios = result.model_specific.get("aspect_ratios").unwrap();
        if let ModelSpecificValue::IntTensor { data, .. } = aspect_ratios {
            let h_tiles = data[0] as usize;
            let w_tiles = data[1] as usize;
            let num_tiles = h_tiles * w_tiles;

            if num_tiles > 1 {
                // 4D output: [total_tiles, C, H, W]
                // total_tiles = num_tiles + 1 (global tile)
                let shape = result.encoder_input.shape();
                assert_eq!(shape[0], num_tiles + 1);
            }
        }
    }

    #[test]
    fn test_model_name() {
        let processor = Llama4VisionProcessor::new();
        assert_eq!(processor.model_name(), "llama4-vision");
    }

    #[test]
    fn test_normalization_values() {
        let processor = Llama4VisionProcessor::new();
        assert_eq!(processor.default_mean(), [0.5, 0.5, 0.5]);
        assert_eq!(processor.default_std(), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_token_count_calculation() {
        let processor = Llama4VisionProcessor::new();
        // 1x1 tile: 576 tokens
        assert_eq!(processor.calculate_num_tokens_for_aspect_ratio((1, 1)), 576);
        // 2x2 tiles + 1 global: 5 * 576 = 2880 tokens
        assert_eq!(
            processor.calculate_num_tokens_for_aspect_ratio((2, 2)),
            2880
        );
        // 1x2 tiles + 1 global: 3 * 576 = 1728 tokens
        assert_eq!(
            processor.calculate_num_tokens_for_aspect_ratio((1, 2)),
            1728
        );
    }

    #[test]
    fn an_empty_batch_is_refused_as_empty_batch() {
        let processor = Llama4VisionProcessor::new();
        let result = processor.preprocess(&[]);
        assert!(matches!(result, Err(TransformError::EmptyBatch)));
        assert!(matches!(
            processor.preprocess_wire(&[], EncoderDtype::BFloat16),
            Some(Err(TransformError::EmptyBatch))
        ));
    }

    /// The one-pass wire against the two-pass bytes (`preprocess`, then the
    /// write in the wire's dtype) in every float dtype: a batch of a canvas
    /// with margins, an exact fit and a single tile.
    #[test]
    fn the_wire_equals_the_two_pass_bytes() {
        let processor = Llama4VisionProcessor::new();
        let image = |width, height| {
            DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |x, y| {
                Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
            }))
        };
        let images = [image(1024, 768), image(672, 336), image(200, 300)];
        let refs: Vec<&DynamicImage> = images.iter().collect();
        let two_pass = processor.preprocess(&refs).unwrap();
        for dtype in [
            EncoderDtype::BFloat16,
            EncoderDtype::Float16,
            EncoderDtype::Float32,
        ] {
            let wire = processor.preprocess_wire(&refs, dtype).unwrap().unwrap();
            assert_eq!(wire.encoder_input.shape(), two_pass.encoder_input.shape());
            assert!(
                wire.encoder_input.write_as(dtype) == two_pass.encoder_input.write_as(dtype),
                "{dtype}: the one-pass wire differs from the two-pass bytes"
            );
            assert_eq!(wire.feature_token_counts, two_pass.feature_token_counts);
            for (key, value) in &two_pass.model_specific {
                assert_eq!(
                    format!("{:?}", wire.model_specific[key]),
                    format!("{value:?}")
                );
            }
        }
    }

    /// `tile_blocks` against the canvas read pixel by pixel: the image's byte
    /// inside it, black (0) beyond it, no stale byte in a partial tile behind a
    /// full one (no pixel byte here is 0).
    #[test]
    fn the_tile_blocks_are_the_canvas_pixel_by_pixel() {
        let (width, height, tile, (rows, columns)) = (7, 7, 3, (3, 4));
        let rgb: Vec<u8> = (1..=251).cycle().take(width * height * 3).collect();
        let mut got = Vec::new();
        tile_blocks(
            &rgb,
            (width, height),
            (rows, columns),
            tile,
            |channel, bytes| {
                got.push((channel, bytes.to_vec()));
            },
        );
        let mut want = Vec::new();
        for tile_y in 0..rows {
            for tile_x in 0..columns {
                for channel in 0..3 {
                    let block: Vec<u8> = (0..tile * tile)
                        .map(|i| (tile_x * tile + i % tile, tile_y * tile + i / tile))
                        .map(|(x, y)| {
                            if x < width && y < height {
                                rgb[(y * width + x) * 3 + channel]
                            } else {
                                0
                            }
                        })
                        .collect();
                    want.push((channel, block));
                }
            }
        }
        assert_eq!(got, want);
    }

    /// The block buffer of `tile_blocks` is the pool's and goes back to it,
    /// and no byte of it reaches `run` unwritten: a debug build hands the
    /// buffer out filled with the pool's sentinel.
    #[test]
    fn the_tile_blocks_buffer_goes_back_to_the_pool() {
        static MINE: buffer_pool::BufferPool = buffer_pool::BufferPool::new();
        let tile = 336;
        let (width, height) = (tile + 64, tile);
        let rgb = vec![1u8; width * height * 3];
        let mut blocks = Vec::new();
        buffer_pool::with_pool(&MINE, || {
            tile_blocks(&rgb, (width, height), (1, 2), tile, |_, bytes| {
                blocks.push(bytes.to_vec());
            });
        });
        // The full tile: every byte the image's.
        assert!(blocks[..3].iter().flatten().all(|&byte| byte == 1));
        // The partial tile: 64 image columns, then black.
        for row in blocks[3..].iter().flat_map(|block| block.chunks(tile)) {
            assert!(row[..64].iter().all(|&byte| byte == 1));
            assert!(row[64..].iter().all(|&byte| byte == 0));
        }
        // Three channels of a 336x336 tile, 338,688 bytes: the 512 KiB class.
        assert_eq!(MINE.held(), 512 << 10);
    }

    /// A tile too large for any allocation, and a batch of them too large
    /// for `usize` to count: the request reads an error in both forms of the
    /// output.
    #[test]
    fn a_patch_output_that_cannot_be_reserved_is_an_error() {
        let processor = Llama4VisionProcessor {
            tile_size: 1 << 30,
            max_patches: 1,
            ..Llama4VisionProcessor::new()
        };
        let image = create_test_image(1, 1, Rgb([0, 0, 0]));
        let wire = |images: &[&DynamicImage]| {
            processor
                .preprocess_wire(images, EncoderDtype::BFloat16)
                .expect("a narrow wire")
        };

        for result in [
            wire(&[&image]),
            wire(&[&image; 3]),
            processor.preprocess(&[&image]),
        ] {
            assert!(
                matches!(&result, Err(TransformError::ReserveRefused(message))
                    if message
                        .strip_prefix("cannot reserve Llama 4 patch output: ")
                        .is_some_and(|cause| !cause.is_empty())),
                "{result:?}"
            );
        }
    }
}
