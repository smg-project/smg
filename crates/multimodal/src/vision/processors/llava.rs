//! LLaVA family image processors.
//!
//! This module implements preprocessing for:
//! - LLaVA 1.5: CLIP-based preprocessing with configurable aspect ratio handling
//! - LLaVA-NeXT: Multi-crop anyres processing for higher resolution
//!
//! # Image Aspect Ratio Modes
//!
//! The processing behavior depends on the `image_aspect_ratio` config:
//!
//! - **None/Square**: Standard CLIP processing (resize shortest edge, center crop)
//! - **"pad"**: Expand to square with mean color padding, then resize
//! - **"anyres"**: Multi-crop processing for higher resolution (LLaVA-NeXT)
//!
//! # Processing Pipeline
//!
//! ## LLaVA 1.5 (Standard - no expand_to_square)
//! Used for `llava-hf/*` models where `image_aspect_ratio` is not set:
//! 1. Resize so shortest edge = target_size (preserving aspect ratio)
//! 2. Center crop to target_size x target_size
//! 3. Rescale by 1/255
//! 4. Normalize with CLIP mean/std
//!
//! ## LLaVA 1.5 (Pad mode - with expand_to_square)
//! Used for `liuhaotian/llava-*` models where `image_aspect_ratio = "pad"`:
//! 1. Expand image to square by padding with mean color
//! 2. Resize to target size (typically 336x336)
//! 3. Normalize with CLIP mean/std
//!
//! ## LLaVA-NeXT
//! 1. Select best resolution from grid pinpoints
//! 2. Resize and pad to best resolution
//! 3. Divide into crops
//! 4. Process each crop + original resized image
//! 5. Stack all processed patches

use std::borrow::Cow;

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{self, Array3};
use serde_json::Value;

use super::qwen_vl_base::pixel_norm;
use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::{
        center_crop, crop_rgb, expand_to_square, mean_to_rgb, paste_rgb, pil_to_filter, resize,
        resize_bicubic_torch_aa, resize_bicubic_torch_aa_cropped, stack_batch, to_tensor_by_table,
        TransformError,
    },
};

/// CLIP normalization mean values used by LLaVA models.
pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization std values used by LLaVA models.
pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// The reference processor's default grid pinpoints, `[[336, 672], [672, 336],
/// [672, 672], [1008, 336], [336, 1008]]` as `[height, width]`, here as
/// `(width, height)` in the same order: an image that fits several pinpoints
/// with the same effective and wasted resolution takes the first of them.
const DEFAULT_GRID_PINPOINTS: [(u32, u32); 5] =
    [(672, 336), (336, 672), (672, 672), (336, 1008), (1008, 336)];

/// Image aspect ratio handling mode.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ImageAspectRatio {
    /// Standard CLIP processing: resize shortest edge, center crop.
    /// Used for llava-hf/* models where image_aspect_ratio is not set.
    #[default]
    Square,
    /// Expand to square with mean color padding, then resize.
    /// Used for liuhaotian/llava-* models where image_aspect_ratio = "pad".
    Pad,
    /// Multi-crop anyres processing (handled by LlavaNextProcessor).
    Anyres,
}

impl std::str::FromStr for ImageAspectRatio {
    type Err = std::convert::Infallible;

    /// Parse from string value (e.g., from config).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "pad" => Self::Pad,
            "anyres" => Self::Anyres,
            _ if s.contains("anyres") => Self::Anyres, // anyres_max_12, etc.
            _ => Self::Square,
        })
    }
}

/// Resolved image transforms shared by the LLaVA variants.
#[derive(Debug, Clone)]
struct LlavaTransforms {
    mean: [f64; 3],
    filter: FilterType,
    target_size: Option<(u32, u32)>,
    crop_size: Option<(u32, u32)>,
    do_resize: bool,
    do_center_crop: bool,
    /// The `f32` each byte maps to, per channel: the reference's rescale and
    /// normalize in its own order and rounding (`pixel_norm`), built once.
    table: [[f32; 256]; 3],
}

impl LlavaTransforms {
    fn new(config: &PreProcessorConfig) -> Self {
        Self {
            mean: config.get_image_mean(),
            // The reference's processors default to bicubic when the config
            // omits `resample`; the crate-wide fallback is bilinear.
            filter: pil_to_filter(config.resampling.or(Some(3))),
            target_size: config.get_target_size(),
            crop_size: config.get_crop_size(),
            do_resize: config.do_resize.unwrap_or(true),
            do_center_crop: config.do_center_crop.unwrap_or(true),
            // Only the table is read here; the channel block describes a
            // patch layout and does not enter it.
            table: pixel_norm(config, CLIP_MEAN, CLIP_STD, 1).table_f32(),
        }
    }
}

/// LLaVA 1.5 image processor.
///
/// Implements CLIP-based preprocessing with configurable aspect ratio handling.
/// This processor is used for LLaVA 1.5 and similar models that expect fixed-size
/// square inputs.
///
/// # Aspect Ratio Modes
///
/// - `Square` (default): Standard CLIP processing (llava-hf/*)
/// - `Pad`: Expand to square with mean padding (liuhaotian/llava-*)
///
/// # Token Calculation
///
/// For LLaVA 1.5, the number of image tokens is:
/// ```text
/// num_tokens = (image_size / patch_size)²
/// ```
/// With default settings (336x336, patch_size=14): 576 tokens
#[derive(Debug, Clone)]
pub struct LlavaProcessor {
    transforms: LlavaTransforms,
    /// Patch size for token calculation (typically 14)
    pub patch_size: u32,
    /// Target image size after processing (typically 336)
    pub image_size: u32,
    /// Image aspect ratio handling mode
    pub aspect_ratio: ImageAspectRatio,
}

impl Default for LlavaProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl LlavaProcessor {
    /// Create a new LLaVA 1.5 processor with default settings.
    ///
    /// Default: patch_size=14, image_size=336, aspect_ratio=Square
    /// This matches the llava-hf/* model behavior.
    pub fn new() -> Self {
        Self {
            transforms: LlavaTransforms::new(&PreProcessorConfig::default()),
            patch_size: 14,
            image_size: 336,
            aspect_ratio: ImageAspectRatio::Square,
        }
    }

    /// Create a processor with "pad" aspect ratio mode.
    ///
    /// This matches the liuhaotian/llava-* model behavior where
    /// images are expanded to square before processing.
    pub fn new_with_pad() -> Self {
        Self {
            transforms: LlavaTransforms::new(&PreProcessorConfig::default()),
            patch_size: 14,
            image_size: 336,
            aspect_ratio: ImageAspectRatio::Pad,
        }
    }

    /// Create a processor with custom settings.
    pub fn with_config(patch_size: u32, image_size: u32, aspect_ratio: ImageAspectRatio) -> Self {
        Self {
            transforms: LlavaTransforms::new(&PreProcessorConfig::default()),
            patch_size,
            image_size,
            aspect_ratio,
        }
    }

    /// Create a processor from model config JSON.
    ///
    /// Extracts patch_size, image_size, and image_aspect_ratio from config.
    pub fn from_config(config: &Value) -> Self {
        let patch_size = config
            .get("vision_config")
            .and_then(|v| v.get("patch_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(14);

        let image_size = config
            .get("vision_config")
            .and_then(|v| v.get("image_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(336);

        let aspect_ratio = config
            .get("image_aspect_ratio")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();

        Self {
            transforms: LlavaTransforms::new(&PreProcessorConfig::default()),
            patch_size,
            image_size,
            aspect_ratio,
        }
    }

    /// Resolve model geometry and image transforms once for a loaded model.
    pub fn from_configs(model_config: &Value, config: &PreProcessorConfig) -> Self {
        Self::from_config(model_config).with_preprocessor_config(config)
    }

    /// Resolve the image transforms and the token patch size from `config`,
    /// keeping this processor's model geometry otherwise.
    pub fn with_preprocessor_config(mut self, config: &PreProcessorConfig) -> Self {
        self.patch_size = config.get_patch_size(self.patch_size as usize) as u32;
        self.transforms = LlavaTransforms::new(config);
        self
    }

    /// Process a single image through the LLaVA 1.5 pipeline.
    ///
    /// The processing flow depends on `self.aspect_ratio`:
    /// - `Square`: Standard CLIP (resize shortest edge, center crop)
    /// - `Pad`: Expand to square with mean padding, then resize
    fn process_one_image(&self, image: &DynamicImage) -> Array3<f32> {
        let transforms = &self.transforms;
        let filter = transforms.filter;

        // Get target size from config or use default
        let target_size = transforms
            .target_size
            .map(|(h, _w)| h)
            .unwrap_or(self.image_size);

        // Get crop size (may be different from target_size)
        let crop_size = transforms.crop_size.map(|(h, _w)| h).unwrap_or(target_size);

        let processed = match self.aspect_ratio {
            ImageAspectRatio::Pad => {
                // Pad mode: expand to square with mean color padding, then resize
                let (w, h) = image.dimensions();
                let squared = if w == h {
                    Cow::Borrowed(image)
                } else {
                    let mean_color = mean_to_rgb(&transforms.mean);
                    Cow::Owned(expand_to_square(image, mean_color))
                };

                // Resize to target size (maintaining square)
                if transforms.do_resize {
                    Cow::Owned(
                        resize_as_reference(&squared, target_size, target_size, filter)
                            .into_owned(),
                    )
                } else {
                    squared
                }
            }
            ImageAspectRatio::Square | ImageAspectRatio::Anyres => {
                // Square mode: Standard CLIP processing
                // 1. Resize so shortest edge = target_size (preserving aspect ratio)
                // 2. Center crop to crop_size x crop_size
                let resized = if transforms.do_resize {
                    let (new_w, new_h) = shortest_edge_size(image.dimensions(), target_size);
                    if transforms.do_center_crop
                        && filter == FilterType::CatmullRom
                        && (new_w > crop_size || new_h > crop_size)
                    {
                        // The crop's pixels alone on x86-64 (the columns a
                        // center crop discards are never resampled, and of the
                        // rows only the ones its height taps read are); the
                        // whole and a crop elsewhere. The same bytes as the
                        // resize followed by `center_crop`.
                        let crop = (crop_size.min(new_w), crop_size.min(new_h));
                        let origin = ((new_w - crop.0) / 2, (new_h - crop.1) / 2);
                        let cropped =
                            resize_bicubic_torch_aa_cropped(image, new_w, new_h, origin, crop);
                        return to_tensor_by_table(&cropped, &transforms.table);
                    }
                    resize_as_reference(image, new_w, new_h, filter)
                } else {
                    Cow::Borrowed(image)
                };

                // Center crop to crop_size (skip if image already fits)
                if transforms.do_center_crop {
                    let (rw, rh) = resized.dimensions();
                    if crop_size >= rw && crop_size >= rh {
                        resized
                    } else {
                        Cow::Owned(center_crop(&resized, crop_size, crop_size))
                    }
                } else {
                    resized
                }
            }
        };

        // [C, H, W] floats: the reference's rescale and normalize of every byte.
        to_tensor_by_table(&processed, &transforms.table)
    }
}

impl VisionPreProcessor for LlavaProcessor {
    fn default_mean(&self) -> [f64; 3] {
        CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        // Store original sizes
        let item_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();

        // Process each image
        let tensors: Vec<Array3<f32>> = images
            .iter()
            .map(|img| self.process_one_image(img))
            .collect();

        // Stack into batch
        let encoder_input = stack_batch(&tensors)?;

        // Calculate token counts
        let feature_token_counts: Vec<usize> = images
            .iter()
            .map(|_| self.calculate_num_tokens(self.image_size, self.image_size))
            .collect::<Result<_, _>>()?;

        Ok(PreprocessedEncoderInputs::new(
            encoder_input,
            feature_token_counts,
            item_sizes,
        ))
    }

    fn calculate_num_tokens(&self, _width: u32, _height: u32) -> Result<usize, TransformError> {
        // For LLaVA 1.5, token count is based on processed image size and patch size
        let patch_size = self.patch_size;
        let image_size = self
            .transforms
            .target_size
            .map(|(h, _w)| h)
            .unwrap_or(self.image_size);

        let patches_per_side = image_size / patch_size;
        Ok((patches_per_side * patches_per_side) as usize)
    }

    fn model_name(&self) -> &'static str {
        "llava"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        let size = self
            .transforms
            .target_size
            .map(|(h, _w)| h)
            .unwrap_or(self.image_size);
        Some((size, size))
    }
}

// ============================================================================
// LLaVA-NeXT (Anyres) Support
// ============================================================================

/// LLaVA-NeXT image processor with anyres (multi-crop) support.
///
/// LLaVA-NeXT processes high-resolution images by:
/// 1. Selecting the best resolution from predefined grid pinpoints
/// 2. Resizing and padding the image to that resolution
/// 3. Dividing into crops
/// 4. Processing each crop plus the original resized image
///
/// # Token Calculation
///
/// For LLaVA-NeXT, the number of tokens depends on the selected resolution:
/// ```text
/// base_tokens = (image_size / patch_size)²
/// grid_shape = (best_width / patch_size, best_height / patch_size)
/// unpad_shape = adjusted for aspect ratio
/// total_tokens = base_tokens + (unpad_w + 1) * unpad_h
/// ```
#[derive(Debug, Clone)]
pub struct LlavaNextProcessor {
    /// Base processor for individual patches
    pub base: LlavaProcessor,
    /// Grid pinpoints for resolution selection [(width, height), ...]
    pub image_grid_pinpoints: Vec<(u32, u32)>,
}

impl Default for LlavaNextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl LlavaNextProcessor {
    /// Resolve model geometry and image transforms once for a loaded model.
    pub fn from_configs(model_config: &Value, config: &PreProcessorConfig) -> Self {
        Self::from_config(model_config).with_preprocessor_config(config)
    }

    /// Resolve the image transforms from `config`, keeping this processor's
    /// model geometry.
    pub fn with_preprocessor_config(mut self, config: &PreProcessorConfig) -> Self {
        self.base.transforms = LlavaTransforms::new(config);
        self
    }

    /// Create a new LLaVA-NeXT processor with default settings.
    ///
    /// Default grid pinpoints are common LLaVA-NeXT resolutions.
    pub fn new() -> Self {
        Self {
            base: LlavaProcessor::new(),
            image_grid_pinpoints: DEFAULT_GRID_PINPOINTS.to_vec(),
        }
    }

    /// Create a processor with custom grid pinpoints.
    pub fn with_grid_pinpoints(grid_pinpoints: Vec<(u32, u32)>) -> Self {
        Self {
            base: LlavaProcessor::new(),
            image_grid_pinpoints: grid_pinpoints,
        }
    }

    /// Create a processor from model config.
    pub fn from_config(config: &Value) -> Self {
        let base = LlavaProcessor::from_config(config);

        let grid_pinpoints = config
            .get("image_grid_pinpoints")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|p| {
                        let pair = p.as_array()?;
                        // HF config stores pinpoints as [height, width]
                        let h = pair.first()?.as_u64()? as u32;
                        let w = pair.get(1)?.as_u64()? as u32;
                        Some((w, h))
                    })
                    .collect()
            })
            .unwrap_or_else(|| DEFAULT_GRID_PINPOINTS.to_vec());

        Self {
            base,
            image_grid_pinpoints: grid_pinpoints,
        }
    }

    /// Select the best resolution from grid pinpoints for the given image.
    ///
    /// Minimizes wasted pixels while maximizing effective resolution.
    pub fn select_best_resolution(&self, original_size: (u32, u32)) -> (u32, u32) {
        select_best_resolution(original_size, &self.image_grid_pinpoints)
    }

    /// Get the grid shape (in crops) for anyres processing.
    ///
    /// Returns (grid_width, grid_height) — the number of 336×336 crops along
    /// each axis of the best-fit resolution.
    pub fn get_anyres_grid_shape(&self, image_size: (u32, u32)) -> (u32, u32) {
        let (width, height) = self.select_best_resolution(image_size);
        (width / self.base.image_size, height / self.base.image_size)
    }

    /// Calculate unpad dimensions based on original aspect ratio.
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the processor; keeps API consistent"
    )]
    pub fn calculate_unpad(&self, grid_shape: (u32, u32), original_size: (u32, u32)) -> (u32, u32) {
        calculate_unpad(grid_shape, original_size)
    }

    /// Resize and pad image to target resolution, maintaining aspect ratio.
    fn resize_and_pad_image(&self, image: &DynamicImage, target: (u32, u32)) -> DynamicImage {
        resize_and_pad_image(image, target, self.base.transforms.filter)
    }

    /// Divide image into crops of specified size.
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the processor; keeps API consistent"
    )]
    fn divide_to_samples(&self, image: &DynamicImage, crop_size: (u32, u32)) -> Vec<DynamicImage> {
        divide_to_samples(image, crop_size)
    }

    /// Process a single patch/crop.
    fn process_patch(&self, image: &DynamicImage) -> Array3<f32> {
        let transforms = &self.base.transforms;
        let filter = transforms.filter;

        // Get target size for patches
        let target_size = transforms
            .target_size
            .map(|(h, _w)| h)
            .unwrap_or(self.base.image_size);

        // Resize patch to target size (a crop of that size stays as it is)
        let resized = if transforms.do_resize {
            resize_as_reference(image, target_size, target_size, filter)
        } else {
            Cow::Borrowed(image)
        };

        // Center crop if configured (skip if image already fits)
        let cropped = if transforms.do_center_crop {
            if let Some((crop_h, crop_w)) = transforms.crop_size {
                let (rw, rh) = resized.dimensions();
                if crop_w >= rw && crop_h >= rh {
                    resized
                } else {
                    Cow::Owned(center_crop(&resized, crop_w, crop_h))
                }
            } else {
                resized
            }
        } else {
            resized
        };

        to_tensor_by_table(&cropped, &transforms.table)
    }
}

impl VisionPreProcessor for LlavaNextProcessor {
    fn default_mean(&self) -> [f64; 3] {
        CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let mut patches_per_image: Vec<Vec<Array3<f32>>> = Vec::with_capacity(images.len());
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());

        let filter = self.base.transforms.filter;
        let target_size = self
            .base
            .transforms
            .target_size
            .map(|(h, _w)| h)
            .unwrap_or(self.base.image_size);
        let crop_size = self
            .base
            .transforms
            .crop_size
            .unwrap_or((target_size, target_size));

        for image in images {
            let original_size = image.dimensions();
            item_sizes.push(original_size);

            let best_resolution = self.select_best_resolution(original_size);
            let image_padded = self.resize_and_pad_image(image, best_resolution);
            let image_original_resize =
                resize_as_reference(image, target_size, target_size, filter).into_owned();

            let mut samples = vec![image_original_resize];
            samples.extend(self.divide_to_samples(&image_padded, crop_size));

            let patches: Vec<Array3<f32>> = samples.iter().map(|s| self.process_patch(s)).collect();
            patches_per_image.push(patches);

            feature_token_counts.push(self.calculate_num_tokens(original_size.0, original_size.1)?);
        }

        // Build 5D encoder_input [num_images, max_patches, C, H, W] matching the
        // HF LlavaNextImageProcessor output the engine expects with Batched layout.
        let (c, h, w) = if let Some(first) = patches_per_image.iter().find_map(|p| p.first()) {
            let s = first.shape();
            (s[0], s[1], s[2])
        } else {
            return Err(TransformError::EmptyBatch);
        };

        let num_images = images.len();
        let encoder_input = stack_patches(&patches_per_image, (c, h, w))?;

        // Build model-specific image_sizes tensor as [num_images, 2] in (height, width) order
        // to match the HF LlavaNextImageProcessor output format the engine expects.
        let image_sizes_flat: Vec<i64> = item_sizes
            .iter()
            .flat_map(|&(w, h)| [h as i64, w as i64])
            .collect();

        let mut result =
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes);
        result.model_specific.insert(
            "image_sizes".to_string(),
            ModelSpecificValue::int_2d(image_sizes_flat, num_images, 2),
        );
        Ok(result)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let original_size = (width, height);

        // Use effective geometry from config, falling back to base defaults.
        let image_size = self
            .base
            .transforms
            .target_size
            .map(|(h, _w)| h)
            .unwrap_or(self.base.image_size);
        let patch_size = self.base.patch_size;

        // Base feature tokens (from original resized image through ViT).
        // CLIP ViT produces (image_size/patch_size)^2 + 1 tokens (patches + CLS).
        // With vision_feature_select_strategy="default" CLS is removed,
        // leaving (image_size/patch_size)^2 = 576 features.
        let npatches = image_size / patch_size; // 24 for 336/14
        let base_features = (npatches * npatches) as usize; // 576

        // Grid shape in crops (e.g. 1x2 for 336x672 best resolution)
        let (grid_w, grid_h) = self.get_anyres_grid_shape(original_size);

        // The unpadded feature dimensions as the reference's
        // `_get_unpadded_features` computes them: feature space
        // (npatches * grid_dim) per axis, the ratios in double precision, the
        // unpadded side rounded to seven decimals and truncated. Rounding it
        // moved the padding by one row on some sizes (1366x768: 1948 tokens
        // against the reference's 1850), and the engine refuses a request
        // whose placeholders do not match its own count.
        let current_w = f64::from(npatches * grid_w);
        let current_h = f64::from(npatches * grid_h);
        let aspect_ratio = f64::from(width) / f64::from(height);
        let current_aspect = current_w / current_h;

        let (feat_h, feat_w) = if aspect_ratio > current_aspect {
            // Landscape: the height is unpadded.
            let new_h = unpadded_side(f64::from(height) * (current_w / f64::from(width)));
            let padding = ((current_h - new_h) / 2.0).floor();
            (current_h - 2.0 * padding, current_w)
        } else {
            // Portrait or square: the width is unpadded.
            let new_w = unpadded_side(f64::from(width) * (current_h / f64::from(height)));
            let padding = ((current_w - new_w) / 2.0).floor();
            (current_h, current_w - 2.0 * padding)
        };

        let unpadded_features = (feat_h * feat_w) as usize;
        let newline_features = feat_h as usize; // one newline token per row

        Ok(unpadded_features + newline_features + base_features)
    }

    fn model_name(&self) -> &'static str {
        "llava-next"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        // LLaVA-NeXT has variable output size based on crops
        // Return the base patch size
        let size = self
            .base
            .transforms
            .target_size
            .map(|(h, _w)| h)
            .unwrap_or(self.base.image_size);
        Some((size, size))
    }
}

// ============================================================================
// Helper Functions (ported from mistral.rs)
// ============================================================================

/// Select the best resolution from possible resolutions for the given image size.
///
/// Minimizes wasted pixels while maximizing effective resolution.
fn select_best_resolution(
    original_size: (u32, u32),
    possible_resolutions: &[(u32, u32)],
) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let mut best_fit = (0, 0);
    // The reference's arithmetic: double precision, truncation to integers.
    let original_width_f = f64::from(original_width);
    let original_height_f = f64::from(original_height);
    let mut max_effective_resolution = 0_u32;
    let mut min_wasted_resolution = u32::MAX;

    for &(width, height) in possible_resolutions {
        let scale =
            (f64::from(width) / original_width_f).min(f64::from(height) / original_height_f);
        let (downscaled_width, downscaled_height) = (
            (original_width_f * scale) as u32,
            (original_height_f * scale) as u32,
        );
        let effective_resolution = std::cmp::min(
            downscaled_width * downscaled_height,
            original_width * original_height,
        );
        let wasted_resolution = width * height - effective_resolution;

        if effective_resolution > max_effective_resolution
            || (effective_resolution == max_effective_resolution
                && wasted_resolution < min_wasted_resolution)
        {
            best_fit = (width, height);
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
        }
    }
    best_fit
}

/// The reference's `int(round(x, 7))` for an unpadded side: rounded to seven
/// decimals, then truncated, so that a product one ulp under an integer counts
/// as that integer and anything else rounds down.
fn unpadded_side(value: f64) -> f64 {
    ((value * 1e7).round() / 1e7).floor()
}

/// Calculate unpad dimensions based on aspect ratio.
fn calculate_unpad(size: (u32, u32), original_size: (u32, u32)) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let (current_width, current_height) = size;
    let original_aspect_ratio = original_width as f32 / original_height as f32;
    let current_aspect_ratio = current_width as f32 / current_height as f32;

    if original_aspect_ratio > current_aspect_ratio {
        let scale_factor = current_width as f32 / original_width as f32;
        let new_height = (original_height as f32 * scale_factor).floor() as u32;
        let padding = (current_height - new_height) / 2;
        (current_width, current_height - 2 * padding)
    } else {
        let scale_factor = current_height as f32 / original_height as f32;
        let new_width = (original_width as f32 * scale_factor).floor() as u32;
        let padding = (current_width - new_width) / 2;
        (current_width - 2 * padding, current_height)
    }
}

/// Resize and pad image to target resolution, centering the image.
fn resize_and_pad_image(
    image: &DynamicImage,
    target: (u32, u32),
    filter: FilterType,
) -> DynamicImage {
    let (original_width, original_height) = image.dimensions();
    let (target_width, target_height) = target;

    // The reference's arithmetic: double precision, the ceiling to an integer.
    let scale_w = f64::from(target_width) / f64::from(original_width);
    let scale_h = f64::from(target_height) / f64::from(original_height);

    let (new_width, new_height) = if scale_w < scale_h {
        (
            target_width,
            std::cmp::min(
                (f64::from(original_height) * scale_w).ceil() as u32,
                target_height,
            ),
        )
    } else {
        (
            std::cmp::min(
                (f64::from(original_width) * scale_h).ceil() as u32,
                target_width,
            ),
            target_height,
        )
    };

    let resized = resize_as_reference(image, new_width, new_height, filter);
    let paste_x = (target_width - new_width) / 2;
    let paste_y = (target_height - new_height) / 2;

    // The paste onto the black canvas copies the stored colours: an alpha
    // channel is dropped the way the reference's RGB conversion drops it,
    // whatever the image's size (the reference resize hands a same-size
    // image back as it is), so no layout blends over the canvas.
    let rgb = match &*resized {
        DynamicImage::ImageRgb8(rgb) => Cow::Borrowed(rgb),
        other => Cow::Owned(other.to_rgb8()),
    };
    DynamicImage::ImageRgb8(paste_rgb(
        &rgb,
        target_width,
        target_height,
        paste_x,
        paste_y,
        [0, 0, 0],
    ))
}

/// The size of the reference's shortest-edge resize: the shorter side becomes
/// `size` and the longer side `size * long / short` truncated to an integer
/// (the reference's `int()`), not rounded. One pixel on the long side moves
/// every sample of the resize and the center crop with it.
fn shortest_edge_size((width, height): (u32, u32), size: u32) -> (u32, u32) {
    let (short, long) = if width <= height {
        (width, height)
    } else {
        (height, width)
    };
    let long = ((u64::from(size) * u64::from(long)) as f64 / f64::from(short)) as u32;
    if width <= height {
        (size, long)
    } else {
        (long, size)
    }
}

/// The reference's resize of a decoded image to `width` x `height`: the image
/// itself when it already has that size (the reference returns it untouched),
/// the torchvision `uint8` antialiased bicubic for a bicubic `filter` (the
/// kernel the family's reference processors run on the bytes, which the SIMD
/// Catmull-Rom parts from by one to ten levels on a tenth of the pixels) and
/// the SIMD resizer for the other filters.
fn resize_as_reference(
    image: &DynamicImage,
    width: u32,
    height: u32,
    filter: FilterType,
) -> Cow<'_, DynamicImage> {
    if image.dimensions() == (width, height) {
        return Cow::Borrowed(image);
    }
    Cow::Owned(if filter == FilterType::CatmullRom {
        resize_bicubic_torch_aa(image, width, height)
    } else {
        resize(image, width, height, filter)
    })
}

/// Divide image into crops of specified size.
fn divide_to_samples(image: &DynamicImage, crop_size: (u32, u32)) -> Vec<DynamicImage> {
    let (width, height) = image.dimensions();
    let mut samples = Vec::new();

    for y in (0..height).step_by(crop_size.1 as usize) {
        for x in (0..width).step_by(crop_size.0 as usize) {
            let patch = match image {
                // Row copies out of the image's own bytes; the generic crop
                // fills a zeroed image pixel by pixel.
                DynamicImage::ImageRgb8(rgb) => {
                    DynamicImage::ImageRgb8(crop_rgb(rgb, x, y, crop_size.0, crop_size.1))
                }
                other => other.crop_imm(x, y, crop_size.0, crop_size.1),
            };
            samples.push(patch);
        }
    }
    samples
}

/// The batch tensor `[num_images, max_patches, c, h, w]`, `max_patches` the
/// widest image's patch count: every image's patches in order, then zero
/// patches up to that width, every element written once (a zeroed tensor
/// assigned patch by patch paid the allocator's page faults for a zero fill
/// the copies then overwrote). A patch of another shape is refused rather
/// than broadcast into its slot.
fn stack_patches(
    patches_per_image: &[Vec<Array3<f32>>],
    (c, h, w): (usize, usize, usize),
) -> Result<ndarray::Array5<f32>, TransformError> {
    let num_images = patches_per_image.len();
    let max_patches = patches_per_image.iter().map(Vec::len).max().unwrap_or(0);
    let patch_len = c * h * w;
    let mut data = Vec::with_capacity(num_images * max_patches * patch_len);
    for patches in patches_per_image {
        for patch in patches {
            if patch.shape() != [c, h, w] {
                return Err(TransformError::InvalidShape {
                    expected: format!("[{c}, {h}, {w}]"),
                    actual: patch.shape().to_vec(),
                });
            }
            match patch.as_slice() {
                Some(values) => data.extend_from_slice(values),
                None => data.extend(patch.iter()),
            }
        }
        data.resize(data.len() + (max_patches - patches.len()) * patch_len, 0.0);
    }
    #[expect(
        clippy::expect_used,
        reason = "data holds num_images * max_patches * c * h * w elements by construction"
    )]
    let tensor = ndarray::Array5::from_shape_vec((num_images, max_patches, c, h, w), data)
        .expect("patch tensor matches its shape");
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::{
        encoder_inputs::EncoderDtype,
        test_support::{create_test_image, hex, lcg_rgb_image},
    };

    #[test]
    fn test_llava_processor_default() {
        let processor = LlavaProcessor::new();
        assert_eq!(processor.patch_size, 14);
        assert_eq!(processor.image_size, 336);
        assert_eq!(processor.aspect_ratio, ImageAspectRatio::Square);
    }

    #[test]
    fn test_llava_processor_with_pad() {
        let processor = LlavaProcessor::new_with_pad();
        assert_eq!(processor.patch_size, 14);
        assert_eq!(processor.image_size, 336);
        assert_eq!(processor.aspect_ratio, ImageAspectRatio::Pad);
    }

    #[test]
    fn test_llava_token_calculation() {
        let processor = LlavaProcessor::new();

        // 336 / 14 = 24, 24 * 24 = 576
        let tokens = processor.calculate_num_tokens(336, 336).unwrap();
        assert_eq!(tokens, 576);
    }

    #[test]
    fn test_llava_preprocess_square() {
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_center_crop: Some(true),
            do_normalize: Some(true),
            image_mean: Some(CLIP_MEAN.to_vec()),
            image_std: Some(CLIP_STD.to_vec()),
            ..Default::default()
        };
        let processor = LlavaProcessor::new().with_preprocessor_config(&config);

        let image = create_test_image(336, 336, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[&image]).unwrap();

        assert_eq!(result.batch_size(), 1);
        assert_eq!(result.height().unwrap(), 336);
        assert_eq!(result.width().unwrap(), 336);
        assert_eq!(result.feature_token_counts[0], 576);
    }

    #[test]
    fn test_llava_preprocess_rectangular_square_mode() {
        // Square mode (default): resize shortest edge, center crop
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_center_crop: Some(true),
            do_normalize: Some(true),
            size: Some([("shortest_edge".to_string(), 336)].into_iter().collect()),
            crop_size: Some(
                [("height".to_string(), 336), ("width".to_string(), 336)]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        };
        let processor = LlavaProcessor::new().with_preprocessor_config(&config);

        // Tall image - should be resized so shortest edge = 336, then center cropped
        let image = create_test_image(200, 400, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[&image]).unwrap();

        assert_eq!(result.batch_size(), 1);
        assert_eq!(result.height().unwrap(), 336);
        assert_eq!(result.width().unwrap(), 336);
    }

    #[test]
    fn test_llava_preprocess_rectangular_pad_mode() {
        // Pad mode: expand to square with mean padding, then resize
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_center_crop: Some(false),
            do_normalize: Some(true),
            ..Default::default()
        };
        let processor = LlavaProcessor::new_with_pad().with_preprocessor_config(&config);

        // Tall image should be padded to square first
        let image = create_test_image(200, 400, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[&image]).unwrap();

        assert_eq!(result.batch_size(), 1);
        // After expand_to_square: 400x400, then resize to 336x336
        assert_eq!(result.height().unwrap(), 336);
        assert_eq!(result.width().unwrap(), 336);
    }

    #[test]
    fn test_select_best_resolution() {
        let pinpoints = vec![(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)];

        // Square image should pick square resolution
        let best = select_best_resolution((500, 500), &pinpoints);
        assert_eq!(best, (672, 672));

        // Wide image should pick wide resolution
        let best = select_best_resolution((800, 400), &pinpoints);
        assert_eq!(best, (672, 336));

        // Tall image should pick tall resolution
        let best = select_best_resolution((400, 800), &pinpoints);
        assert_eq!(best, (336, 672));
    }

    #[test]
    fn test_calculate_unpad() {
        // Square grid, wide original -> should reduce width padding
        let unpad = calculate_unpad((24, 24), (800, 400));
        assert!(unpad.0 >= unpad.1); // Width should be >= height

        // Square grid, tall original -> should reduce height padding
        let unpad = calculate_unpad((24, 24), (400, 800));
        assert!(unpad.1 >= unpad.0); // Height should be >= width
    }

    #[test]
    fn test_llava_next_processor_default() {
        let processor = LlavaNextProcessor::new();
        assert!(!processor.image_grid_pinpoints.is_empty());
        assert_eq!(processor.base.patch_size, 14);
    }

    #[test]
    fn test_llava_next_preprocess() {
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_center_crop: Some(false),
            do_normalize: Some(true),
            ..Default::default()
        };
        let processor = LlavaNextProcessor::new().with_preprocessor_config(&config);

        let image = create_test_image(500, 500, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[&image]).unwrap();

        // 5D: [num_images=1, num_patches, C, H, W]
        assert_eq!(result.batch_size(), 1);
        // Should have multiple patches (original + crops) in the second dimension
        assert!(result.encoder_input.shape()[1] > 1);
    }

    #[test]
    fn test_divide_to_samples() {
        let image = create_test_image(672, 672, Rgb([128, 128, 128]));
        let samples = divide_to_samples(&image, (336, 336));

        // 672x672 / 336x336 = 2x2 = 4 patches
        assert_eq!(samples.len(), 4);

        for sample in &samples {
            assert_eq!(sample.width(), 336);
            assert_eq!(sample.height(), 336);
        }
    }

    /// An image whose bytes vary with the position: a misplaced byte shows.
    fn gradient(width: u32, height: u32, seed: usize) -> RgbImage {
        RgbImage::from_raw(
            width,
            height,
            crate::test_support::gradient_rgb(width as usize, height as usize, seed),
        )
        .expect("gradient size")
    }

    #[test]
    fn the_padded_image_is_the_generic_paste_on_black() {
        // A wide image (padded above and below) and a tall one (padded on
        // the sides), each resized the way the processor resizes it and
        // pasted with the generic paste onto a zeroed canvas.
        for (width, height, target) in [(100u32, 40u32, (672u32, 336u32)), (40, 100, (336, 672))] {
            let image = DynamicImage::ImageRgb8(gradient(width, height, 3));
            let padded = resize_and_pad_image(&image, target, FilterType::CatmullRom);
            let (scale_w, scale_h) = (
                target.0 as f64 / width as f64,
                target.1 as f64 / height as f64,
            );
            let (new_w, new_h) = if scale_w < scale_h {
                (
                    target.0,
                    ((height as f64 * scale_w).ceil() as u32).min(target.1),
                )
            } else {
                (
                    ((width as f64 * scale_h).ceil() as u32).min(target.0),
                    target.1,
                )
            };
            let resized = resize_as_reference(&image, new_w, new_h, FilterType::CatmullRom);
            let mut expected = DynamicImage::new_rgb8(target.0, target.1);
            image::imageops::overlay(
                &mut expected,
                &*resized,
                i64::from((target.0 - new_w) / 2),
                i64::from((target.1 - new_h) / 2),
            );
            assert_eq!(padded.dimensions(), (target.0, target.1));
            assert_eq!(padded.as_bytes(), expected.as_bytes(), "{width}x{height}");
        }
    }

    #[test]
    fn an_image_with_alpha_is_pasted_by_its_stored_colours_at_every_size() {
        // A same-size image comes back from the reference resize as it is, a
        // resized one as RGB: both canvases hold the stored colours, the alpha
        // dropped as the reference's RGB conversion drops it, not blended over
        // the black canvas.
        for (width, height) in [(672u32, 672u32), (673, 672)] {
            let rgba = image::RgbaImage::from_fn(width, height, |x, y| {
                image::Rgba([(x % 256) as u8, (y % 256) as u8, 200, 0])
            });
            let opaque = DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(rgba.clone()).to_rgb8());
            let padded = resize_and_pad_image(
                &DynamicImage::ImageRgba8(rgba),
                (672, 672),
                FilterType::CatmullRom,
            );
            let expected = resize_and_pad_image(&opaque, (672, 672), FilterType::CatmullRom);
            assert_eq!(padded.as_bytes(), expected.as_bytes(), "{width}x{height}");
        }
    }

    #[test]
    fn the_crops_are_the_generic_crops() {
        // A grid of whole crops, a grid the image's edges clip, a column.
        for (width, height) in [(672, 672), (700, 500), (336, 1008)] {
            let image = DynamicImage::ImageRgb8(gradient(width, height, 5));
            let samples = divide_to_samples(&image, (336, 336));
            let mut expected = Vec::new();
            for y in (0..height).step_by(336) {
                for x in (0..width).step_by(336) {
                    expected.push(image.crop_imm(x, y, 336, 336));
                }
            }
            assert_eq!(samples.len(), expected.len(), "{width}x{height}");
            for (sample, expected) in samples.iter().zip(&expected) {
                assert_eq!(
                    sample.dimensions(),
                    expected.dimensions(),
                    "{width}x{height}"
                );
                assert_eq!(sample.as_bytes(), expected.as_bytes(), "{width}x{height}");
            }
        }
    }

    #[test]
    fn a_patch_of_another_shape_is_refused_not_broadcast() {
        // A degenerate `[3, 336, 1]` patch: a zeroed tensor's assign would
        // have broadcast it across its slot; the copy refuses it.
        let patches = vec![vec![Array3::<f32>::zeros((3, 336, 1))]];
        assert!(matches!(
            stack_patches(&patches, (3, 336, 336)),
            Err(TransformError::InvalidShape { .. })
        ));
    }

    #[test]
    fn the_batch_tensor_holds_every_patch_and_zero_padding() {
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_center_crop: Some(false),
            do_normalize: Some(true),
            ..Default::default()
        };
        let processor = LlavaNextProcessor::new().with_preprocessor_config(&config);
        // A square image takes the square grid (four crops and the base
        // image); a small wide one a two-crop grid, padded to the square's
        // patch count in the batch.
        let square = DynamicImage::ImageRgb8(gradient(500, 500, 7));
        let wide = DynamicImage::ImageRgb8(gradient(100, 50, 11));
        let result = processor.preprocess(&[&square, &wide]).unwrap();
        let tensor = result.encoder_input.as_f32();
        assert_eq!(tensor.shape(), &[2, 5, 3, 336, 336]);

        // The zeroed form: a zero tensor with every patch assigned.
        let filter = processor.base.transforms.filter;
        let mut expected = ndarray::Array5::<f32>::zeros((2, 5, 3, 336, 336));
        for (i, image) in [&square, &wide].into_iter().enumerate() {
            let best_resolution = processor.select_best_resolution(image.dimensions());
            let padded = processor.resize_and_pad_image(image, best_resolution);
            let mut samples = vec![resize_as_reference(image, 336, 336, filter).into_owned()];
            samples.extend(processor.divide_to_samples(&padded, (336, 336)));
            for (j, sample) in samples.iter().enumerate() {
                expected
                    .slice_mut(ndarray::s![i, j, .., .., ..])
                    .assign(&processor.process_patch(sample));
            }
        }
        assert_eq!(tensor.as_slice().unwrap(), expected.as_slice().unwrap());
    }

    /// The checkpoint's `preprocessor_config.json` the reference ran with.
    fn reference_config() -> PreProcessorConfig {
        PreProcessorConfig::from_json(
            r#"{"crop_size": {"height": 336, "width": 336}, "do_center_crop": true,
                "do_convert_rgb": true, "do_normalize": true, "do_rescale": true,
                "do_resize": true,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "resample": 3, "rescale_factor": 0.00392156862745098,
                "size": {"shortest_edge": 336}}"#,
        )
        .unwrap()
    }

    fn f32_at(bytes: &[u8], index: usize) -> f32 {
        f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
    }

    /// A pinned case: a seeded image, the SHA-256 of the reference's f32
    /// tensor on the same bytes, and four of its elements as
    /// `(index, the value the previous code gave, the reference's value)`.
    struct Pinned {
        width: u32,
        height: u32,
        seed: u32,
        digest: &'static str,
        witnesses: &'static [(usize, f32, f32)],
    }

    fn assert_pinned(bytes: &[u8], case: &Pinned) {
        use sha2::{Digest, Sha256};

        let Pinned {
            width,
            height,
            digest,
            witnesses,
            ..
        } = case;
        for &(index, previous, reference) in *witnesses {
            let value = f32_at(bytes, index);
            assert!(
                value.to_bits() == reference.to_bits(),
                "{width}x{height} element {index}: {value} (the previous code gave {previous}, \
                 the reference {reference})"
            );
        }
        assert_eq!(hex(&Sha256::digest(bytes)), *digest, "{width}x{height}");
    }

    /// The reference's shortest-edge resize truncates the long side (`int()`);
    /// rounding it, as the processor did, moved every sample of a 47x22 image.
    #[test]
    fn shortest_edge_resize_truncates_the_long_side_as_the_reference() {
        assert_eq!(shortest_edge_size((47, 22), 336), (717, 336)); // 717.8: not 718
        assert_eq!(shortest_edge_size((22, 47), 336), (336, 717));
        assert_eq!(shortest_edge_size((1024, 768), 336), (448, 336));
        assert_eq!(shortest_edge_size((336, 336), 336), (336, 336));
    }

    /// The reference's processors default to bicubic when a config omits
    /// `resample`, and the crate-wide fallback is bilinear: a processor built
    /// without a config (`new()`, `with_grid_pinpoints`) resized, padded and
    /// patched through the other kernel.
    #[test]
    fn a_config_without_resample_resizes_bicubic_like_the_reference() {
        let bicubic = LlavaTransforms::new(&PreProcessorConfig::default());
        assert_eq!(bicubic.filter, FilterType::CatmullRom);
        let bilinear = PreProcessorConfig {
            resampling: Some(2),
            ..Default::default()
        };
        assert_eq!(LlavaTransforms::new(&bilinear).filter, FilterType::Triangle);
    }

    /// The anyres token count truncates the unpadded side as the reference's
    /// `_get_unpadded_features` does (`int(round(x, 7))`); rounding it moved
    /// the padding by one row on the first three sizes (default pinpoints,
    /// a 336 image of 14-pixel patches).
    #[test]
    fn anyres_token_counts_truncate_the_unpadded_side_as_the_reference() {
        let processor = LlavaNextProcessor::new();
        for (width, height, tokens) in [
            (1366, 768, 1850),  // rounded: 1948
            (2560, 1080, 1944), // rounded: 1992
            (100, 31, 1262),    // rounded: 1360
            (1920, 1080, 1948),
            (1024, 768, 2340),
        ] {
            assert_eq!(
                processor.calculate_num_tokens(width, height).unwrap_or(0),
                tokens,
                "{width}x{height}"
            );
        }
    }

    /// A solid image goes through no resize and no crop, so every element is
    /// the table's entry for its byte: the reference's
    /// `(byte - mean * 255) / (std * 255)` in `f32`. The previous
    /// `(byte / 255 - mean) * (1 / std)` landed one ULP away on most levels
    /// (its bits in the comment column).
    #[test]
    fn the_byte_table_carries_the_reference_arithmetic() {
        // (byte level, the reference's f32 bits per channel)
        let levels: [(u8, [u32; 3]); 4] = [
            (0, [0xbfe5_68dc, 0xbfe0_44b7, 0xbfbd_77d8]), // was bfe568db bfe044b8 bfbd77d7
            (1, [0xbfe3_8a80, 0xbfde_58f1, 0xbfbb_a5e2]), // was bfe38a7f bfde58f1 bfbba5e1
            (128, [0x3d9c_560e, 0x3e2c_f360, 0x3eae_0dc1]), // was 3d9c561f 3e2cf36b 3eae0dc4
            (255, [0x3ff7_1542, 0x4004_cae4, 0x4009_5661]), // was 3ff71541 4004cae5 40095660
        ];
        let processor = LlavaProcessor::new().with_preprocessor_config(&reference_config());
        let plane = 336 * 336;
        for (level, bits) in levels {
            let image = create_test_image(336, 336, Rgb([level, level, level]));
            let result = processor.preprocess(&[&image]).unwrap();
            let bytes = result.encoder_input.write_as(EncoderDtype::Float32);
            assert_eq!(bytes.len(), 3 * plane * 4);
            for (channel, expected) in bits.into_iter().enumerate() {
                let differing = (channel * plane..(channel + 1) * plane)
                    .filter(|&index| f32_at(&bytes, index).to_bits() != expected)
                    .count();
                assert_eq!(
                    differing,
                    0,
                    "level {level} channel {channel}: {} elements are not {:e}",
                    differing,
                    f32::from_bits(expected)
                );
            }
        }
    }

    /// Two seeded images through the LLaVA 1.5 pipeline (shortest edge to
    /// 336, the long side truncated, center crop, the byte table) against
    /// the reference processor's output on the same bytes. The previous
    /// kernel and arithmetic differed on 98 % of the elements, by up to half
    /// a normalized unit (34 levels).
    #[test]
    fn square_pixels_match_the_reference_bit_for_bit() {
        let cases = [
            Pinned {
                width: 47,
                height: 22,
                seed: 17,
                digest: "689379d9448235ebb241c2469828e6ca07bb76fd2085e463a0cb03680b5e7734",
                witnesses: &[
                    (0, 1.3755962, 1.3609976),
                    (1, 1.3901947, 1.3755959),
                    (168_469, 0.108866245, 0.13888161),
                    (338_687, 1.5202143, 1.5059942),
                ],
            },
            Pinned {
                width: 22,
                height: 47,
                seed: 19,
                digest: "93820fb925930e91c5352aaae675674154c2cce424534a0ad3a53d82d42745b7",
                witnesses: &[
                    (0, 0.36830464, 0.38290295),
                    (1, 0.3537062, 0.36830452),
                    (168_207, 1.9998451, 1.9848369),
                    (338_687, -0.04399298, -0.029773047),
                ],
            },
        ];
        let processor = LlavaProcessor::new().with_preprocessor_config(&reference_config());
        for case in &cases {
            let image = lcg_rgb_image(case.width, case.height, case.seed);
            let result = processor.preprocess(&[&image]).unwrap();
            assert_eq!(result.encoder_input.shape(), &[1, 3, 336, 336]);
            assert_pinned(&result.encoder_input.write_as(EncoderDtype::Float32), case);
        }
    }

    /// Two seeded images through the anyres pipeline (the base image plus the
    /// two crops of a 336x672 or 672x336 best fit, zero padding between)
    /// against the reference processor's output on the same bytes.
    #[test]
    fn anyres_patches_match_the_reference_bit_for_bit() {
        let cases = [
            Pinned {
                width: 50,
                height: 80,
                seed: 23,
                digest: "0a3e2fce52f497bbb9924bf4c53c3f86de8a802dde9b92fb571e9247e36eec18",
                witnesses: &[
                    (0, 1.3901947, 1.3609976),
                    (1, 1.404793, 1.3755959),
                    (515_366, -0.041211426, 0.003811711),
                    (1_016_063, -1.4802197, -1.4802198),
                ],
            },
            Pinned {
                width: 90,
                height: 40,
                seed: 29,
                digest: "5d3e3201a87eff97701f05e6776916b659837e5b4bdbc10f51401bb25d3007ba",
                witnesses: &[
                    (0, 0.10553299, 0.120131284),
                    (1, 0.0033440161, 0.0033438762),
                    (504_902, 0.67916137, 0.67916125),
                    (1_016_063, -1.4802197, -1.4802198),
                ],
            },
        ];
        let processor = LlavaNextProcessor::new().with_preprocessor_config(&reference_config());
        for case in &cases {
            let image = lcg_rgb_image(case.width, case.height, case.seed);
            let result = processor.preprocess(&[&image]).unwrap();
            assert_eq!(result.encoder_input.shape(), &[1, 3, 3, 336, 336]);
            assert_pinned(&result.encoder_input.write_as(EncoderDtype::Float32), case);
        }
    }
}
