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
    pixel_convert::{Bf16Steps, StepsWriter},
    vision::{
        execution,
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

    /// The reference processor's arithmetic: it rescales by `1 / 255` and
    /// normalizes in `bfloat16`, one rounded step after another, whatever
    /// dtype the engine then casts to. Every float this processor emits is
    /// that table's.
    fn steps(&self) -> Bf16Steps {
        Bf16Steps {
            factor: 1.0 / 255.0,
            mean: self.mean.map(|mean| mean as f32),
            std: self.std.map(|std| std as f32),
        }
    }

    /// `(byte - 255 * mean) / (255 * std)` in `f32`, the fused order of
    /// [`transforms::to_tensor_and_normalize`], in the form a wire carries:
    /// the nearest a `PixelNorm` comes to [`Self::steps`], whose values the
    /// wire holds (at the default mean and std the two are within half a
    /// byte step of each other).
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
        let global = (self.tile_size, self.tile_size);
        let (resized, view) = if image.dimensions() == plan.resized {
            // An image already at its size is its own bytes, and it has no
            // empty side: a planned size has none.
            let (_, _, rgb) = transforms::rgb_bytes(image);
            let view = plan.has_global().then(|| {
                execution::on_caller(|| {
                    transforms::resize_bilinear_torch_aa_rgb_bytes(
                        &rgb, width, height, global.0, global.1,
                    )
                })
            });
            (rgb, view)
        } else if plan.has_global() {
            let [resized, view] = resized_rgb(image, [plan.resized, global]);
            (Cow::Owned(resized), Some(view))
        } else {
            let [resized] = resized_rgb(image, [plan.resized]);
            (Cow::Owned(resized), None)
        };
        let size = (width as usize, height as usize);
        tile_blocks(&resized, size, plan.grid, tile, &mut run);
        // The resized bytes are the pool's: back they go behind their tiles.
        if let Cow::Owned(resized) = resized {
            buffer_pool::recycle(resized);
        }
        if let Some(view) = view {
            tile_blocks(&view, (tile, tile), (1, 1), tile, &mut run);
            buffer_pool::recycle(view);
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

/// `image`'s RGB bytes resized to each of `targets`. The reference resizes
/// the decoded RGB bytes with torchvision's `uint8` BILINEAR. An image with
/// an empty side has no pixel to resample and keeps the generic resizer. So
/// does a colour image of more than four megapixels on a processor whose
/// width pass runs its AVX2 rows, where the exact kernel trails that resizer
/// ([`transforms::bilinear_torch_aa_trails_generic`]): there the image keeps
/// the bytes it had before this resize was exact, not the reference's. The
/// resize stays on the caller's thread, where this family has always
/// resized: its bands on the pool would save wall time and cost CPU time.
fn resized_rgb<const N: usize>(image: &DynamicImage, targets: [(u32, u32); N]) -> [Vec<u8>; N] {
    let empty = image.width() == 0 || image.height() == 0;
    if empty || transforms::bilinear_torch_aa_trails_generic(image) {
        return targets.map(|(width, height)| {
            match transforms::resize(image, width, height, FilterType::Triangle) {
                DynamicImage::ImageRgb8(rgb) => rgb.into_raw(),
                resized => transforms::rgb_bytes(&resized).2.into_owned(),
            }
        });
    }
    execution::on_caller(|| transforms::resize_bilinear_torch_aa_image(image, targets))
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
        let table = self.steps().table_f32();
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
        let writer = StepsWriter::new(&self.steps(), dtype)?;
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
            EncoderInput::wire(wire, dtype, vec![tiles, 3, tile, tile], self.norm())
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
    use image::{GrayImage, Rgb, RgbaImage};

    use super::*;
    use crate::test_support::{assert_same_bytes, create_test_image, lcg_bytes, lcg_rgb_image};

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

    /// The tile bytes `write_image` hands on for `image` at a tile of 8.
    fn tile_bytes(image: &DynamicImage) -> Vec<u8> {
        tile_bytes_within(image, DEFAULT_MAX_PATCHES)
    }

    /// [`tile_bytes`] on a canvas of at most `max_patches` tiles.
    fn tile_bytes_within(image: &DynamicImage, max_patches: usize) -> Vec<u8> {
        let processor = Llama4VisionProcessor {
            tile_size: 8,
            max_patches,
            ..Llama4VisionProcessor::new()
        };
        let mut bytes = Vec::new();
        let plan = processor.plan(image.dimensions());
        processor.write_image(image, &plan, |_, channel| bytes.extend_from_slice(channel));
        bytes
    }

    /// A resized image's tiles hold the reference processor's bytes:
    /// torchvision's `resize` of the `uint8` image (bilinear, antialiased), on
    /// generator bytes at a tile of 8. A 6x4 image is scaled up to 8x5 inside
    /// its one tile; a 13x7 and a 7x13 image fill two tiles as they are, and
    /// the global view of each is the image scaled to 8x8, the one down its
    /// width, the other down its height. The bytes are what the reference
    /// class (transformers 5.17, torch 2.13, torchvision 0.28) hands its
    /// normalize step for the same images. To make them again: build the
    /// class with `size={"height": 8, "width": 8}`, call it on the image and
    /// read the `uint8` tensors it hands `rescale_and_normalize`, the canvas
    /// and then the global view.
    #[test]
    fn a_resized_image_holds_the_reference_bytes() {
        let scaled_up: [u8; 192] = [
            68, 174, 153, 17, 42, 20, 64, 162, 123, 137, 153, 165, 158, 140, 147, 170, 116, 87, 87,
            128, 198, 161, 141, 154, 123, 87, 49, 41, 151, 111, 107, 159, 209, 166, 115, 72, 63,
            68, 125, 216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            50, 162, 174, 91, 148, 215, 190, 91, 93, 114, 98, 69, 179, 227, 203, 133, 117, 117,
            103, 97, 213, 206, 148, 78, 141, 127, 129, 156, 222, 154, 95, 74, 187, 95, 104, 209,
            196, 97, 113, 237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 84, 63, 114, 221, 223, 90, 31, 66, 174, 95, 105, 201, 217, 166, 130, 125, 199, 113,
            119, 212, 197, 194, 167, 122, 147, 130, 159, 217, 173, 161, 133, 92, 58, 164, 214, 186,
            160, 95, 69, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,
        ];
        let wide_global_view: [u8; 192] = [
            107, 111, 83, 54, 182, 186, 193, 90, 141, 142, 140, 129, 163, 93, 134, 118, 143, 195,
            145, 141, 113, 82, 138, 125, 154, 152, 177, 176, 131, 91, 122, 130, 155, 124, 145, 181,
            155, 143, 116, 122, 134, 159, 120, 165, 164, 179, 107, 89, 134, 163, 216, 183, 197,
            134, 76, 74, 223, 126, 143, 113, 162, 194, 174, 178, 180, 82, 186, 191, 82, 145, 73,
            41, 118, 132, 152, 106, 157, 134, 71, 64, 162, 193, 120, 138, 169, 145, 162, 163, 194,
            195, 144, 131, 110, 116, 156, 183, 144, 159, 173, 83, 99, 120, 124, 149, 109, 130, 175,
            77, 148, 120, 119, 132, 155, 106, 163, 134, 175, 71, 100, 152, 6, 38, 112, 130, 208,
            242, 129, 189, 128, 126, 90, 104, 126, 77, 119, 235, 114, 84, 88, 127, 64, 118, 171,
            227, 123, 39, 95, 94, 64, 120, 137, 94, 151, 50, 86, 128, 63, 117, 97, 132, 165, 96,
            110, 125, 50, 84, 94, 195, 133, 128, 146, 85, 64, 71, 126, 152, 68, 114, 132, 113, 117,
            138, 147, 105, 63, 53, 151, 87, 138, 143, 75, 167,
        ];
        let tall_global_view: [u8; 192] = [
            76, 18, 128, 170, 131, 113, 161, 180, 159, 143, 144, 156, 172, 163, 111, 80, 148, 142,
            96, 121, 181, 182, 117, 138, 131, 128, 81, 59, 138, 227, 173, 79, 74, 106, 113, 92,
            129, 197, 191, 145, 139, 154, 181, 139, 148, 189, 148, 187, 167, 104, 162, 156, 134,
            135, 124, 152, 54, 41, 108, 184, 159, 67, 51, 114, 129, 100, 159, 205, 189, 135, 121,
            213, 167, 110, 116, 87, 84, 91, 48, 133, 162, 102, 149, 111, 104, 126, 69, 131, 144,
            85, 98, 68, 53, 84, 100, 92, 142, 152, 171, 139, 105, 116, 141, 130, 155, 107, 124,
            167, 183, 164, 142, 157, 156, 126, 133, 166, 148, 104, 123, 162, 124, 164, 163, 130,
            111, 90, 33, 32, 104, 165, 173, 148, 107, 93, 144, 191, 71, 171, 196, 128, 86, 123,
            171, 194, 148, 148, 162, 169, 124, 105, 198, 223, 168, 154, 123, 153, 193, 174, 109,
            110, 104, 155, 133, 153, 173, 121, 53, 152, 62, 135, 110, 127, 113, 51, 58, 165, 109,
            123, 145, 101, 107, 149, 127, 222, 132, 196, 144, 120, 162, 186, 127, 107,
        ];
        assert_same_bytes(
            &tile_bytes(&lcg_rgb_image(6, 4, 21)),
            &scaled_up,
            "6x4 scaled up to 8x5",
        );
        let wide = tile_bytes(&lcg_rgb_image(13, 7, 22));
        assert_same_bytes(
            &wide[wide.len() - 192..],
            &wide_global_view,
            "the global view of 13x7",
        );
        let tall = tile_bytes(&lcg_rgb_image(7, 13, 25));
        assert_same_bytes(
            &tall[tall.len() - 192..],
            &tall_global_view,
            "the global view of 7x13",
        );
    }

    /// An image larger than every canvas takes the arm a large photo takes:
    /// it is scaled down on both axes into more than one tile, and its global
    /// view is the image itself scaled to one tile. A 37x21 image of
    /// generator bytes on a canvas of at most two tiles is scaled to 14x8;
    /// the bytes are the reference class's for the same image (the recipe of
    /// [`a_resized_image_holds_the_reference_bytes`] with `max_patches=2`):
    /// its two tiles, then its global view. Pillow's bilinear taps give
    /// other bytes for this image, on the canvas and on the global view.
    #[test]
    fn an_image_scaled_down_into_two_tiles_holds_the_reference_bytes() {
        let tiles_and_global_view: [u8; 576] = [
            156, 165, 145, 133, 144, 135, 126, 129, 105, 97, 124, 125, 100, 109, 103, 153, 141,
            152, 141, 121, 94, 115, 115, 127, 143, 123, 127, 134, 136, 132, 133, 92, 149, 149, 116,
            151, 144, 167, 154, 116, 159, 156, 131, 144, 172, 165, 152, 132, 126, 149, 119, 116,
            115, 137, 140, 153, 135, 128, 102, 117, 163, 121, 125, 142, 102, 107, 96, 139, 137,
            129, 143, 134, 86, 102, 117, 127, 148, 120, 136, 108, 140, 144, 146, 141, 152, 136,
            120, 125, 110, 147, 136, 149, 153, 137, 133, 102, 121, 153, 124, 154, 145, 104, 140,
            131, 139, 148, 114, 104, 116, 102, 103, 144, 120, 125, 114, 95, 133, 135, 109, 127,
            119, 129, 131, 109, 151, 127, 119, 113, 92, 112, 133, 154, 109, 124, 148, 121, 150,
            140, 129, 108, 96, 118, 115, 116, 122, 108, 123, 151, 123, 115, 151, 123, 99, 111, 110,
            112, 125, 110, 110, 128, 113, 135, 85, 119, 135, 144, 116, 146, 141, 136, 137, 126,
            123, 131, 107, 134, 118, 156, 148, 148, 115, 137, 124, 128, 154, 154, 111, 127, 168,
            163, 126, 140, 148, 98, 107, 106, 123, 170, 0, 0, 144, 105, 111, 134, 121, 117, 0, 0,
            146, 131, 118, 111, 104, 99, 0, 0, 107, 108, 149, 132, 136, 115, 0, 0, 108, 93, 120,
            71, 121, 152, 0, 0, 106, 118, 136, 103, 115, 149, 0, 0, 114, 89, 101, 98, 102, 139, 0,
            0, 123, 128, 102, 138, 148, 126, 0, 0, 111, 93, 122, 137, 139, 138, 0, 0, 92, 109, 133,
            117, 107, 144, 0, 0, 98, 108, 135, 119, 145, 111, 0, 0, 124, 145, 161, 119, 116, 141,
            0, 0, 112, 140, 126, 136, 119, 98, 0, 0, 136, 108, 116, 139, 123, 124, 0, 0, 115, 103,
            97, 140, 146, 148, 0, 0, 132, 116, 104, 165, 116, 150, 0, 0, 129, 138, 148, 149, 126,
            142, 0, 0, 109, 136, 129, 123, 147, 150, 0, 0, 127, 112, 77, 102, 125, 129, 0, 0, 151,
            137, 114, 100, 104, 94, 0, 0, 149, 140, 124, 131, 137, 105, 0, 0, 103, 92, 116, 149,
            123, 127, 0, 0, 92, 102, 111, 139, 117, 128, 0, 0, 102, 118, 137, 139, 127, 135, 0, 0,
            162, 143, 139, 132, 135, 109, 109, 147, 103, 121, 109, 108, 147, 111, 126, 117, 149,
            136, 105, 116, 134, 128, 112, 102, 136, 126, 134, 132, 101, 118, 139, 125, 148, 131,
            150, 157, 114, 103, 95, 133, 156, 137, 165, 155, 123, 121, 114, 132, 139, 123, 118,
            140, 136, 95, 98, 123, 132, 109, 143, 126, 135, 118, 127, 140, 104, 109, 137, 137, 123,
            105, 134, 136, 93, 117, 138, 129, 102, 114, 120, 125, 140, 144, 147, 126, 115, 112,
            126, 131, 126, 142, 151, 132, 113, 147, 129, 130, 134, 138, 140, 124, 125, 130, 131,
            108, 145, 114, 112, 103, 140, 112, 133, 121, 125, 109, 124, 120, 122, 102, 130, 145,
            124, 125, 136, 121, 120, 116, 141, 133, 100, 137, 125, 133, 129, 139, 145, 133, 145,
            126, 102, 116, 113, 129, 129, 146, 112, 130, 129, 132, 127, 105, 100, 124, 102, 112,
            121, 108, 138, 134, 103, 101, 121, 101, 137, 127, 144, 138, 130, 119, 141, 133, 128,
            118, 117, 100, 137, 124, 137, 150, 129, 129, 112, 102, 128, 123, 152, 122, 157, 143,
            123, 119, 137, 130,
        ];
        assert_same_bytes(
            &tile_bytes_within(&lcg_rgb_image(37, 21, 43), 2),
            &tiles_and_global_view,
            "37x21 scaled down to 14x8 in two tiles",
        );
    }

    /// An image with alpha is resized as the reference resizes it: the alpha
    /// is dropped first and the RGB bytes are resampled as they stand, never
    /// weighted by it. A 5x3 RGBA image of generator bytes is scaled up to
    /// 8x4 inside its one tile; the bytes are the reference class's for the
    /// same image.
    #[test]
    fn an_image_with_alpha_is_resized_as_its_rgb_bytes() {
        let scaled_up: [u8; 192] = [
            234, 207, 174, 182, 150, 78, 140, 194, 164, 125, 80, 136, 149, 121, 146, 167, 143, 112,
            77, 136, 160, 149, 141, 136, 179, 173, 167, 182, 177, 152, 127, 110, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 41,
            100, 177, 223, 238, 236, 233, 17, 85, 172, 148, 166, 225, 182, 144, 24, 99, 188, 105,
            109, 200, 151, 105, 21, 77, 142, 62, 72, 172, 152, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 74, 83, 143, 126,
            31, 36, 47, 44, 93, 152, 102, 73, 65, 61, 58, 104, 155, 209, 101, 52, 58, 100, 132,
            232, 237, 235, 142, 69, 14, 140, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let image = RgbaImage::from_raw(5, 3, lcg_bytes(5 * 3 * 4, 27)).expect("image size");
        assert_same_bytes(
            &tile_bytes(&DynamicImage::ImageRgba8(image)),
            &scaled_up,
            "5x3 with alpha scaled up to 8x4",
        );
    }

    /// A grey image is written as its RGB conversion: a 6x4 image, which is
    /// scaled up inside its one tile; a 13x7 image's two tiles (the image as
    /// it is) and global view (the image scaled to 8x8) as those of the
    /// converted image; and so a 700x600 image, which is resized on both
    /// axes.
    #[test]
    fn a_grey_image_is_written_as_its_rgb_conversion() {
        for (width, height) in [(6, 4), (13, 7), (700, 600)] {
            let levels = lcg_bytes((width * height) as usize, 31);
            let grey = GrayImage::from_raw(width, height, levels).expect("image size");
            let grey = DynamicImage::ImageLuma8(grey);
            assert_same_bytes(
                &tile_bytes(&grey),
                &tile_bytes(&DynamicImage::ImageRgb8(grey.to_rgb8())),
                &format!("{width}x{height} grey"),
            );
        }
    }

    /// An image with an empty side has no pixel to resample: it keeps the
    /// generic resizer and writes one black tile.
    #[test]
    fn an_image_with_an_empty_side_writes_a_black_tile() {
        for (width, height) in [(0, 5), (5, 0)] {
            assert_same_bytes(
                &tile_bytes(&DynamicImage::new_rgb8(width, height)),
                &[0; 192],
                &format!("{width}x{height}"),
            );
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

    /// A resized image's bytes, its global view's and the band buffer of
    /// their resizes are the pool's and go back behind the tiles: the pool
    /// holds them all when the image is written. A band buffer that had
    /// grown would hold no class's capacity, and the pool would not keep it.
    #[test]
    fn the_resized_bytes_go_back_to_the_pool() {
        static MINE: buffer_pool::BufferPool = buffer_pool::BufferPool::new();
        let processor = Llama4VisionProcessor::new();
        let image = lcg_rgb_image(1500, 1100, 7);
        let plan = processor.plan(image.dimensions());
        assert!(plan.has_global() && plan.resized != image.dimensions());
        buffer_pool::with_pool(&MINE, || processor.write_image(&image, &plan, |_, _| {}));
        let (width, height) = plan.resized;
        let resized = buffer_pool::allocation_capacity(width as usize * height as usize * 3);
        // The band buffer (512 KiB of rows and a block's rounding: the
        // 1 MiB class, one for both resizes), the view and the tile buffer
        // (338,688 bytes each: the 512 KiB class), the resized bytes. Off
        // x86-64 the resize rows allocate vectors of their exact length,
        // which the pool does not keep: the tile buffer alone is its own.
        let held = if cfg!(target_arch = "x86_64") {
            (1 << 20) + 2 * (512 << 10) + resized
        } else {
            512 << 10
        };
        assert_eq!(MINE.held(), held);
    }

    /// A colour image of more than four megapixels holds the bytes of the
    /// road its processor's rows name: the generic resizer's where the width
    /// pass runs its AVX2 rows, the exact kernel's elsewhere.
    #[test]
    fn a_large_colour_image_takes_the_road_of_its_rows() {
        let (width, height) = (2001, 2000);
        let image = lcg_rgb_image(width, height, 11);
        let targets = [(1344, 1343), (336, 336)];
        let resized = resized_rgb(&image, targets);
        let (_, _, rgb) = transforms::rgb_bytes(&image);
        for (bytes, (out_w, out_h)) in resized.iter().zip(targets) {
            let expected = if transforms::bilinear_torch_aa_trails_generic(&image) {
                let generic = transforms::resize(&image, out_w, out_h, FilterType::Triangle);
                transforms::rgb_bytes(&generic).2.into_owned()
            } else {
                transforms::resize_bilinear_torch_aa_rgb_bytes(&rgb, width, height, out_w, out_h)
            };
            assert_same_bytes(bytes, &expected, "a colour image above four megapixels");
        }
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

    /// The reference processor's words for the byte levels 0 to 255 at the
    /// default mean and std, the three channels alike: its `bfloat16`
    /// tensor. A `float32` reads each word widened.
    const REFERENCE_BF16: [u16; 256] = [
        0xbf80, 0xbf7e, 0xbf7c, 0xbf7a, 0xbf78, 0xbf76, 0xbf74, 0xbf72, 0xbf70, 0xbf6e, 0xbf6c,
        0xbf6a, 0xbf68, 0xbf66, 0xbf64, 0xbf62, 0xbf60, 0xbf5e, 0xbf5c, 0xbf5a, 0xbf58, 0xbf56,
        0xbf54, 0xbf52, 0xbf50, 0xbf4e, 0xbf4c, 0xbf4a, 0xbf48, 0xbf46, 0xbf44, 0xbf42, 0xbf40,
        0xbf3e, 0xbf3c, 0xbf3a, 0xbf38, 0xbf36, 0xbf34, 0xbf32, 0xbf30, 0xbf2e, 0xbf2c, 0xbf2a,
        0xbf28, 0xbf26, 0xbf24, 0xbf22, 0xbf20, 0xbf1e, 0xbf1c, 0xbf1a, 0xbf18, 0xbf16, 0xbf14,
        0xbf12, 0xbf10, 0xbf0e, 0xbf0c, 0xbf0a, 0xbf08, 0xbf06, 0xbf04, 0xbf02, 0xbefe, 0xbefa,
        0xbef6, 0xbef2, 0xbeee, 0xbeea, 0xbee6, 0xbee2, 0xbede, 0xbeda, 0xbed6, 0xbed2, 0xbece,
        0xbeca, 0xbec6, 0xbec2, 0xbebe, 0xbeba, 0xbeb6, 0xbeb2, 0xbeae, 0xbeaa, 0xbea6, 0xbea2,
        0xbe9e, 0xbe9a, 0xbe96, 0xbe92, 0xbe8e, 0xbe8a, 0xbe86, 0xbe82, 0xbe7c, 0xbe74, 0xbe6c,
        0xbe64, 0xbe5c, 0xbe54, 0xbe4c, 0xbe44, 0xbe3c, 0xbe34, 0xbe2c, 0xbe24, 0xbe1c, 0xbe14,
        0xbe0c, 0xbe04, 0xbdf8, 0xbde8, 0xbdd8, 0xbdc8, 0xbdb8, 0xbda8, 0xbd98, 0xbd88, 0xbd70,
        0xbd50, 0xbd30, 0xbd10, 0xbce0, 0xbca0, 0xbc40, 0xbb80, 0x3c00, 0x3c80, 0x3cc0, 0x3d00,
        0x3d20, 0x3d40, 0x3d60, 0x3d80, 0x3d90, 0x3da0, 0x3db0, 0x3dc0, 0x3dd0, 0x3de0, 0x3df0,
        0x3e00, 0x3e08, 0x3e10, 0x3e18, 0x3e20, 0x3e28, 0x3e30, 0x3e38, 0x3e40, 0x3e48, 0x3e50,
        0x3e58, 0x3e60, 0x3e68, 0x3e70, 0x3e78, 0x3e80, 0x3e84, 0x3e88, 0x3e8c, 0x3e90, 0x3e94,
        0x3e98, 0x3e9c, 0x3ea0, 0x3ea4, 0x3ea8, 0x3eac, 0x3eb0, 0x3eb4, 0x3eb8, 0x3ebc, 0x3ec0,
        0x3ec4, 0x3ec8, 0x3ecc, 0x3ed0, 0x3ed4, 0x3ed8, 0x3edc, 0x3ee0, 0x3ee4, 0x3ee8, 0x3eec,
        0x3ef0, 0x3ef4, 0x3ef8, 0x3efc, 0x3f00, 0x3f02, 0x3f04, 0x3f06, 0x3f08, 0x3f0a, 0x3f0c,
        0x3f0e, 0x3f10, 0x3f12, 0x3f14, 0x3f16, 0x3f18, 0x3f1a, 0x3f1c, 0x3f1e, 0x3f20, 0x3f22,
        0x3f24, 0x3f26, 0x3f28, 0x3f2a, 0x3f2c, 0x3f2e, 0x3f30, 0x3f32, 0x3f34, 0x3f36, 0x3f38,
        0x3f3a, 0x3f3c, 0x3f3e, 0x3f40, 0x3f42, 0x3f44, 0x3f46, 0x3f48, 0x3f4a, 0x3f4c, 0x3f4e,
        0x3f50, 0x3f52, 0x3f54, 0x3f56, 0x3f58, 0x3f5a, 0x3f5c, 0x3f5e, 0x3f60, 0x3f62, 0x3f64,
        0x3f66, 0x3f68, 0x3f6a, 0x3f6c, 0x3f6e, 0x3f70, 0x3f72, 0x3f74, 0x3f76, 0x3f78, 0x3f7a,
        0x3f7c, 0x3f7e, 0x3f80,
    ];

    /// [`REFERENCE_BF16`] cast to `float16`, which holds every value exactly.
    const REFERENCE_F16: [u16; 256] = [
        0xbc00, 0xbbf0, 0xbbe0, 0xbbd0, 0xbbc0, 0xbbb0, 0xbba0, 0xbb90, 0xbb80, 0xbb70, 0xbb60,
        0xbb50, 0xbb40, 0xbb30, 0xbb20, 0xbb10, 0xbb00, 0xbaf0, 0xbae0, 0xbad0, 0xbac0, 0xbab0,
        0xbaa0, 0xba90, 0xba80, 0xba70, 0xba60, 0xba50, 0xba40, 0xba30, 0xba20, 0xba10, 0xba00,
        0xb9f0, 0xb9e0, 0xb9d0, 0xb9c0, 0xb9b0, 0xb9a0, 0xb990, 0xb980, 0xb970, 0xb960, 0xb950,
        0xb940, 0xb930, 0xb920, 0xb910, 0xb900, 0xb8f0, 0xb8e0, 0xb8d0, 0xb8c0, 0xb8b0, 0xb8a0,
        0xb890, 0xb880, 0xb870, 0xb860, 0xb850, 0xb840, 0xb830, 0xb820, 0xb810, 0xb7f0, 0xb7d0,
        0xb7b0, 0xb790, 0xb770, 0xb750, 0xb730, 0xb710, 0xb6f0, 0xb6d0, 0xb6b0, 0xb690, 0xb670,
        0xb650, 0xb630, 0xb610, 0xb5f0, 0xb5d0, 0xb5b0, 0xb590, 0xb570, 0xb550, 0xb530, 0xb510,
        0xb4f0, 0xb4d0, 0xb4b0, 0xb490, 0xb470, 0xb450, 0xb430, 0xb410, 0xb3e0, 0xb3a0, 0xb360,
        0xb320, 0xb2e0, 0xb2a0, 0xb260, 0xb220, 0xb1e0, 0xb1a0, 0xb160, 0xb120, 0xb0e0, 0xb0a0,
        0xb060, 0xb020, 0xafc0, 0xaf40, 0xaec0, 0xae40, 0xadc0, 0xad40, 0xacc0, 0xac40, 0xab80,
        0xaa80, 0xa980, 0xa880, 0xa700, 0xa500, 0xa200, 0x9c00, 0x2000, 0x2400, 0x2600, 0x2800,
        0x2900, 0x2a00, 0x2b00, 0x2c00, 0x2c80, 0x2d00, 0x2d80, 0x2e00, 0x2e80, 0x2f00, 0x2f80,
        0x3000, 0x3040, 0x3080, 0x30c0, 0x3100, 0x3140, 0x3180, 0x31c0, 0x3200, 0x3240, 0x3280,
        0x32c0, 0x3300, 0x3340, 0x3380, 0x33c0, 0x3400, 0x3420, 0x3440, 0x3460, 0x3480, 0x34a0,
        0x34c0, 0x34e0, 0x3500, 0x3520, 0x3540, 0x3560, 0x3580, 0x35a0, 0x35c0, 0x35e0, 0x3600,
        0x3620, 0x3640, 0x3660, 0x3680, 0x36a0, 0x36c0, 0x36e0, 0x3700, 0x3720, 0x3740, 0x3760,
        0x3780, 0x37a0, 0x37c0, 0x37e0, 0x3800, 0x3810, 0x3820, 0x3830, 0x3840, 0x3850, 0x3860,
        0x3870, 0x3880, 0x3890, 0x38a0, 0x38b0, 0x38c0, 0x38d0, 0x38e0, 0x38f0, 0x3900, 0x3910,
        0x3920, 0x3930, 0x3940, 0x3950, 0x3960, 0x3970, 0x3980, 0x3990, 0x39a0, 0x39b0, 0x39c0,
        0x39d0, 0x39e0, 0x39f0, 0x3a00, 0x3a10, 0x3a20, 0x3a30, 0x3a40, 0x3a50, 0x3a60, 0x3a70,
        0x3a80, 0x3a90, 0x3aa0, 0x3ab0, 0x3ac0, 0x3ad0, 0x3ae0, 0x3af0, 0x3b00, 0x3b10, 0x3b20,
        0x3b30, 0x3b40, 0x3b50, 0x3b60, 0x3b70, 0x3b80, 0x3b90, 0x3ba0, 0x3bb0, 0x3bc0, 0x3bd0,
        0x3be0, 0x3bf0, 0x3c00,
    ];

    /// The side of [`every_level`]'s tile.
    const TILE: usize = 336;

    /// One tile with no resize whose pixels hold every byte level in every
    /// channel.
    fn every_level() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(
            TILE as u32,
            TILE as u32,
            |x, _| Rgb([x as u8, (x + 85) as u8, (x + 170) as u8]),
        ))
    }

    /// The byte level behind element `index` of the `[1, 3, 336, 336]`
    /// output of [`every_level`].
    fn level(index: usize) -> usize {
        (index % TILE + 85 * (index / (TILE * TILE))) % 256
    }

    /// The bytes of [`every_level`]'s wire in `dtype`.
    fn wire_bytes(dtype: EncoderDtype) -> Vec<u8> {
        let wire = Llama4VisionProcessor::new()
            .preprocess_wire(&[&every_level()], dtype)
            .unwrap()
            .unwrap();
        wire.encoder_input.write_as(dtype).to_vec()
    }

    /// The float tensor holds the reference processor's value for every
    /// byte level.
    #[test]
    fn the_float_tensor_holds_the_reference_values() {
        let floats = Llama4VisionProcessor::new()
            .preprocess(&[&every_level()])
            .unwrap();
        let floats = floats.encoder_input.as_f32();
        assert_eq!(floats.len(), 3 * TILE * TILE);
        for (index, value) in floats.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                u32::from(REFERENCE_BF16[level(index)]) << 16,
                "float tensor, element {index}"
            );
        }
    }

    /// The wire of a two-byte float `dtype` holds `reference`'s word for
    /// every byte level.
    fn two_byte_wire_holds(dtype: EncoderDtype, reference: &[u16; 256]) {
        let bytes = wire_bytes(dtype);
        assert_eq!(bytes.len(), 2 * 3 * TILE * TILE);
        for (index, word) in bytes.as_chunks::<2>().0.iter().enumerate() {
            assert_eq!(
                u16::from_le_bytes(*word),
                reference[level(index)],
                "{dtype} wire, element {index}"
            );
        }
    }

    #[test]
    fn the_bfloat16_wire_holds_the_reference_values() {
        two_byte_wire_holds(EncoderDtype::BFloat16, &REFERENCE_BF16);
    }

    #[test]
    fn the_float16_wire_holds_the_reference_values() {
        two_byte_wire_holds(EncoderDtype::Float16, &REFERENCE_F16);
    }

    #[test]
    fn the_float32_wire_holds_the_reference_values() {
        let bytes = wire_bytes(EncoderDtype::Float32);
        assert_eq!(bytes.len(), 4 * 3 * TILE * TILE);
        for (index, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                u32::from_le_bytes(*word),
                u32::from(REFERENCE_BF16[level(index)]) << 16,
                "float32 wire, element {index}"
            );
        }
    }

    /// The byte levels at which the fused formula
    /// ([`Llama4VisionProcessor::norm`], the order this processor normalized
    /// in before it took the reference's steps) gives another value than the
    /// reference, by float form: half of the 256 levels as `bfloat16`, 240
    /// as `float16` and 254 as `float32`.
    #[test]
    fn the_fused_formula_differs_from_the_reference_by_form() {
        use crate::encoder_inputs::{f32_to_bf16_bits, f32_to_f16_bits};
        let fused = Llama4VisionProcessor::new().norm().table_f32()[0];
        let differing = |same: &dyn Fn(f32, usize) -> bool| {
            (0..256).filter(|&level| !same(fused[level], level)).count()
        };
        let bf16 = differing(&|value, level| f32_to_bf16_bits(value) == REFERENCE_BF16[level]);
        let f16 = differing(&|value, level| f32_to_f16_bits(value) == REFERENCE_F16[level]);
        let f32s =
            differing(&|value, level| value.to_bits() == u32::from(REFERENCE_BF16[level]) << 16);
        assert_eq!((bf16, f16, f32s), (128, 240, 254));
    }
}
