//! Phi3-Vision image processor.
//!
//! This module implements the Phi3-Vision image preprocessing pipeline with
//! Dynamic High Definition (HD) transform, which tiles images into 336x336 crops.
//!
//! # Processing Pipeline
//!
//! 1. **HD Transform**: Resize (Pillow's bilinear, what torchvision's
//!    `resize` runs on a PIL image) and pad image to multiples of 336
//! 2. **Normalize**: `ToTensor` then CLIP `Normalize`, in torchvision's steps
//! 3. **Create Global Image**: torch's bicubic interpolation (no
//!    antialiasing) to 336x336
//! 4. **Tile**: Reshape into (num_tiles, 3, 336, 336)
//! 5. **Concatenate**: [global_image, tiles...]
//! 6. **Pad**: Zero-pad to (num_crops+1, 3, 336, 336)
//!
//! # Key Features
//!
//! - Dynamic resolution via HD transform
//! - Default num_crops: 16
//! - CLIP normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
//! - Token count formula: `((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12`

use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use ndarray::{s, Array3, Array4, IxDyn};

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::{self, TransformError},
};

/// CLIP normalization mean values.
pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization std values.
pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Default number of crops for HD transform.
pub const DEFAULT_NUM_CROPS: usize = 16;

/// Default number of image tokens per crop (144 per tile + base).
pub const DEFAULT_NUM_IMG_TOKENS: usize = 144;

/// Tile size used in Phi3-Vision (336x336).
pub const TILE_SIZE: u32 = 336;

/// Phi3-Vision image processor.
///
/// Implements Dynamic HD transform with tile-based processing.
#[derive(Debug, Clone)]
pub struct Phi3VisionProcessor {
    /// Maximum number of HD crops (not including global).
    num_crops: usize,
    /// Normalization mean.
    mean: [f64; 3],
    /// Normalization std.
    std: [f64; 3],
}

impl Default for Phi3VisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Phi3VisionProcessor {
    /// Create a new Phi3-Vision processor with default settings.
    pub fn new() -> Self {
        Self {
            num_crops: DEFAULT_NUM_CROPS,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
    }

    /// Create a processor with custom settings.
    pub fn with_config(num_crops: usize) -> Self {
        Self {
            num_crops,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
    }

    /// This processor normalized with `config`'s mean and std where the config
    /// sets them; the crop budget stays this processor's own.
    pub fn with_normalization_of(mut self, config: &PreProcessorConfig) -> Self {
        if let Some(mean) = &config.image_mean {
            self.mean = [mean[0], mean[1], mean[2]];
        }
        if let Some(std) = &config.image_std {
            self.std = [std[0], std[1], std[2]];
        }
        self
    }

    /// Create a processor from preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            num_crops: config.num_crops.unwrap_or(DEFAULT_NUM_CROPS),
            mean: config
                .image_mean
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(CLIP_MEAN),
            std: config
                .image_std
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(CLIP_STD),
        }
    }

    /// Get the number of crops.
    pub fn num_crops(&self) -> usize {
        self.num_crops
    }

    /// The HD transform's target size for an image of `width` x `height`
    /// in landscape orientation (the transform turns a portrait image on its
    /// side first): the scale is the largest whose tiles fit the crop budget,
    /// the width `scale * 336`, the height that width over the aspect ratio.
    /// Both sides must have a pixel: the scale loop does not end on a zero
    /// side (`hd_target_or_refuse` refuses one before calling this).
    fn hd_target(&self, width: u32, height: u32) -> (u32, u32) {
        let ratio = width as f64 / height as f64;
        let mut scale = 1.0f64;
        while scale * (scale / ratio).ceil() <= self.num_crops as f64 {
            scale += 1.0;
        }
        scale -= 1.0;
        let new_w = (scale * TILE_SIZE as f64) as u32;
        let new_h = (new_w as f64 / ratio) as u32;
        (new_w, new_h)
    }

    /// The HD transform's target for an image as given (either orientation),
    /// refused when it has no pixel on its shorter side (a landscape image's
    /// height, a portrait image's width): an image whose longer side is more
    /// than `num_crops * 336` times its shorter side scales to a target with
    /// a zero side, on which the reference's resize raises rather than
    /// producing a pad-only tile. An image with no pixel on a side to begin
    /// with is refused before the target is computed: the scale loop divides
    /// by the shorter side and would not end on it.
    fn hd_target_or_refuse(&self, width: u32, height: u32) -> Result<(u32, u32), TransformError> {
        if width == 0 || height == 0 {
            return Err(TransformError::ShapeError(format!(
                "a {width}x{height} image has a side with no pixel; the HD transform needs a \
                 pixel on each side"
            )));
        }
        let (long, short) = if width < height {
            (height, width)
        } else {
            (width, height)
        };
        let target = self.hd_target(long, short);
        if target.0 == 0 || target.1 == 0 {
            // The target in the image's own orientation: a portrait image
            // loses its width, not its height.
            let (target_w, target_h) = if width < height {
                (target.1, target.0)
            } else {
                target
            };
            return Err(TransformError::ShapeError(format!(
                "a {width}x{height} image has no pixel on its shorter side after the HD transform \
                 (target {target_w}x{target_h}): an image whose longer side is more than {} times \
                 its shorter side is refused, as the reference's resize refuses it",
                self.num_crops as u32 * TILE_SIZE
            )));
        }
        Ok(target)
    }

    /// The HD transform's output size for an image as given, in the image's
    /// own orientation, from the target alone: the target's height padded up
    /// to whole tiles (`padding_336`; the width is whole tiles already), the
    /// sides swapped back for a portrait image, which the transform turns on
    /// its side and back. The token count needs only this size, not the
    /// transformed image.
    fn hd_output_size(&self, width: u32, height: u32) -> Result<(u32, u32), TransformError> {
        let (target_w, target_h) = self.hd_target_or_refuse(width, height)?;
        let padded_h = target_h.div_ceil(TILE_SIZE) * TILE_SIZE;
        Ok(if width < height {
            (padded_h, target_w)
        } else {
            (target_w, padded_h)
        })
    }

    /// HD transform: resize and pad image to multiples of 336.
    ///
    /// Algorithm:
    /// 1. If width < height, transpose (flip over main diagonal)
    /// 2. Calculate scale: while scale * ceil(scale/ratio) <= hd_num: scale++
    /// 3. Resize to new_w = scale * 336, new_h = new_w / ratio
    /// 4. Pad height to multiple of 336 (centered, white padding)
    /// 5. If transposed, transpose back
    pub fn hd_transform(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();

        let (img, transposed) = if width < height {
            // Transpose (PIL's Image.TRANSPOSE): equivalent to fliph + rotate270 (ccw 90°)
            // This swaps x and y coordinates: pixel at (x, y) goes to (y, x)
            (image.fliph().rotate270(), true)
        } else {
            (image.clone(), false)
        };

        let (width, height) = img.dimensions();
        let (new_w, new_h) = self.hd_target(width, height);

        // The reference resizes the PIL image through torchvision's `resize`
        // at its default interpolation, which is `PIL.Image.resize(BILINEAR)`:
        // Pillow's triangle filter widened by the downscale factor, in its
        // fixed point, two passes.
        let resized = transforms::resize_bilinear_pil(&img, new_w, new_h);

        // Pad height to multiple of 336
        let padded = self.padding_336(&resized);

        // Transpose back if needed (transpose is self-inverse)
        if transposed {
            padded.fliph().rotate270()
        } else {
            padded
        }
    }

    /// Pad image height to multiple of 336 (centered, white padding).
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the processor; keeps API consistent"
    )]
    fn padding_336(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();
        let target_h = ((height as f64 / TILE_SIZE as f64).ceil() * TILE_SIZE as f64) as u32;

        if height == target_h {
            return image.clone();
        }

        let top_padding = (target_h - height) / 2;

        // Create white-padded image
        let mut new_image =
            DynamicImage::from(RgbImage::from_pixel(width, target_h, Rgb([255, 255, 255])));

        // Copy original image to center
        image::imageops::overlay(&mut new_image, image, 0, top_padding as i64);

        new_image
    }

    /// Create global image by bicubic interpolation to 336x336.
    ///
    /// The reference interpolates the normalized tensor with
    /// `torch.nn.functional.interpolate(mode='bicubic')`: no antialiasing,
    /// `align_corners=False`, the `a = -0.75` cubic; `resize_bicubic_torch_f32`
    /// is that kernel bit for bit.
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the processor; keeps API consistent"
    )]
    fn create_global_image(&self, tensor: &Array3<f32>) -> Array3<f32> {
        transforms::resize_bicubic_torch_f32(tensor, TILE_SIZE as usize, TILE_SIZE as usize)
    }

    /// Reshape HD image into tiles.
    ///
    /// Transforms [3, H, W] -> [num_tiles, 3, 336, 336]
    /// where H and W are multiples of 336.
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the processor; keeps API consistent"
    )]
    fn reshape_to_tiles(&self, tensor: &Array3<f32>) -> Vec<Array3<f32>> {
        let (_c, h, w) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
        let grid_h = h / TILE_SIZE as usize;
        let grid_w = w / TILE_SIZE as usize;

        let mut tiles = Vec::with_capacity(grid_h * grid_w);

        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let y_start = gh * TILE_SIZE as usize;
                let x_start = gw * TILE_SIZE as usize;
                let y_end = y_start + TILE_SIZE as usize;
                let x_end = x_start + TILE_SIZE as usize;

                let tile_view = tensor.slice(s![.., y_start..y_end, x_start..x_end]);
                tiles.push(tile_view.to_owned());
            }
        }

        tiles
    }

    /// Calculate number of image tokens for given HD size.
    ///
    /// Formula: `((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12`
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the processor; keeps API consistent"
    )]
    pub fn calculate_num_tokens(&self, h: usize, w: usize) -> usize {
        let grid_h = h / TILE_SIZE as usize;
        let grid_w = w / TILE_SIZE as usize;

        // ((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12
        (grid_h * grid_w + 1) * 144 + 1 + (grid_h + 1) * 12
    }

    /// Process a single image through the full pipeline.
    fn process_single_image(&self, image: &DynamicImage) -> (Array4<f32>, (usize, usize), usize) {
        // 1. Convert to RGB
        let image = DynamicImage::ImageRgb8(image.to_rgb8());

        // 2. HD transform
        let hd_image = self.hd_transform(&image);
        let (hd_w, hd_h) = hd_image.dimensions();

        // 3. To tensor [0, 1] and normalize, in torchvision's steps (the
        //    reference composes `ToTensor` and `Normalize`: three roundings)
        let tensor = transforms::to_tensor_then_normalize(&hd_image, &self.mean, &self.std);

        // 4. Create global image (336x336)
        let global_image = self.create_global_image(&tensor);

        // 5. Reshape HD image into tiles
        let tiles = self.reshape_to_tiles(&tensor);

        // 6. Concatenate global + tiles
        let max_crops = self.num_crops + 1; // num_crops + 1 for global

        // Create output tensor [max_crops, 3, 336, 336]
        let mut output =
            Array4::<f32>::zeros((max_crops, 3, TILE_SIZE as usize, TILE_SIZE as usize));

        // Copy global image (first position)
        output.slice_mut(s![0, .., .., ..]).assign(&global_image);

        // Copy tiles (positions 1..num_actual_crops)
        for (i, tile) in tiles.iter().enumerate() {
            if i + 1 < max_crops {
                output.slice_mut(s![i + 1, .., .., ..]).assign(tile);
            }
        }

        // Calculate token count
        let num_tokens = self.calculate_num_tokens(hd_h as usize, hd_w as usize);

        // The returned size is the HD-transformed image size.
        (output, (hd_h as usize, hd_w as usize), num_tokens)
    }
}

impl VisionPreProcessor for Phi3VisionProcessor {
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
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let mut all_pixel_values = Vec::with_capacity(images.len());
        let mut all_image_sizes = Vec::with_capacity(images.len());
        let mut all_num_tokens = Vec::with_capacity(images.len());

        for image in images {
            let (width, height) = image.dimensions();
            self.hd_target_or_refuse(width, height)?;
            let (encoder_input, image_size, num_tokens) = self.process_single_image(image);
            all_pixel_values.push(encoder_input);
            all_image_sizes.push((image_size.1 as u32, image_size.0 as u32)); // (width, height)
            all_num_tokens.push(num_tokens);
        }

        // Stack into batch [B, num_crops+1, 3, 336, 336]
        let max_crops = self.num_crops + 1;
        let batch_size = images.len();
        let mut batch_tensor = ndarray::Array5::<f32>::zeros((
            batch_size,
            max_crops,
            3,
            TILE_SIZE as usize,
            TILE_SIZE as usize,
        ));

        for (i, pv) in all_pixel_values.iter().enumerate() {
            batch_tensor.slice_mut(s![i, .., .., .., ..]).assign(pv);
        }

        // Convert to dynamic array for storage
        let shape = batch_tensor.shape().to_vec();
        let (flat_data, _offset) = batch_tensor.into_raw_vec_and_offset();

        // Store model-specific image_sizes data.
        let mut model_specific = std::collections::HashMap::new();

        // image_sizes as [batch, 2] tensor (h, w for each image)
        let image_sizes_data: Vec<i64> = all_image_sizes
            .iter()
            .flat_map(|(w, h)| [*h as i64, *w as i64]) // [h, w] for each image
            .collect();
        model_specific.insert(
            "image_sizes".to_string(),
            ModelSpecificValue::IntTensor {
                data: image_sizes_data,
                shape: vec![batch_size, 2],
            },
        );

        // feature_token_counts as list
        model_specific.insert(
            "num_img_tokens".to_string(),
            ModelSpecificValue::IntVec(all_num_tokens.iter().map(|&t| t as i64).collect()),
        );

        // Convert 5D tensor to appropriate format
        // Phi3-Vision expects [B, num_crops+1, C, H, W]
        let encoder_input = ndarray::ArrayD::<f32>::from_shape_vec(IxDyn(&shape), flat_data)
            .map_err(|e| TransformError::InvalidShape {
                expected: format!("valid 5D shape, but failed with error: {e}"),
                actual: shape.clone(),
            })?;

        Ok(PreprocessedEncoderInputs {
            encoder_input: encoder_input.into(),
            feature_token_counts: all_num_tokens,
            item_sizes: all_image_sizes,
            model_specific,
        })
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let (hd_w, hd_h) = self.hd_output_size(width, height)?;
        Ok(self.calculate_num_tokens(hd_h as usize, hd_w as usize))
    }

    fn model_name(&self) -> &'static str {
        "phi3-vision"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        // Phi3-Vision has dynamic size based on HD transform
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::create_test_image;

    #[test]
    fn test_phi3_vision_processor_default() {
        let processor = Phi3VisionProcessor::new();
        assert_eq!(processor.num_crops(), 16);
        assert_eq!(processor.default_mean(), CLIP_MEAN);
        assert_eq!(processor.default_std(), CLIP_STD);
    }

    #[test]
    fn test_hd_transform_square() {
        let processor = Phi3VisionProcessor::new();
        let image = create_test_image(504, 504, Rgb([128, 128, 128]));

        let hd_image = processor.hd_transform(&image);
        let (w, h) = hd_image.dimensions();

        // Should be multiple of 336
        assert_eq!(h % 336, 0);
        assert_eq!(w % 336, 0);

        // Should respect num_crops limit
        let num_tiles = (h / 336) * (w / 336);
        assert!(num_tiles <= processor.num_crops() as u32);
    }

    #[test]
    fn test_hd_transform_tall() {
        let processor = Phi3VisionProcessor::new();
        let image = create_test_image(400, 600, Rgb([100, 100, 100]));

        let hd_image = processor.hd_transform(&image);
        let (w, h) = hd_image.dimensions();

        // Should be multiple of 336
        assert_eq!(h % 336, 0);
        assert_eq!(w % 336, 0);
    }

    #[test]
    fn test_hd_transform_wide() {
        let processor = Phi3VisionProcessor::new();
        let image = create_test_image(600, 400, Rgb([150, 150, 150]));

        let hd_image = processor.hd_transform(&image);
        let (w, h) = hd_image.dimensions();

        // Should be multiple of 336
        assert_eq!(h % 336, 0);
        assert_eq!(w % 336, 0);
    }

    #[test]
    fn test_calculate_num_tokens() {
        let processor = Phi3VisionProcessor::new();

        // 1344x1344 -> 4x4 grid -> (16+1)*144 + 1 + (4+1)*12 = 2448 + 1 + 60 = 2509
        let tokens = processor.calculate_num_tokens(1344, 1344);
        assert_eq!(tokens, 2509);

        // 1008x1344 -> 3x4 grid -> (12+1)*144 + 1 + (3+1)*12 = 1872 + 1 + 48 = 1921
        let tokens = processor.calculate_num_tokens(1008, 1344);
        assert_eq!(tokens, 1921);

        // 1344x1008 -> 4x3 grid -> (12+1)*144 + 1 + (4+1)*12 = 1872 + 1 + 60 = 1933
        let tokens = processor.calculate_num_tokens(1344, 1008);
        assert_eq!(tokens, 1933);
    }

    #[test]
    fn test_phi3_vision_preprocess() {
        let processor = Phi3VisionProcessor::new();

        let image = create_test_image(504, 504, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[&image]).unwrap();

        assert_eq!(result.batch_size(), 1);

        // Check output shape is [1, num_crops+1, 3, 336, 336]
        let shape = result.encoder_input.shape();
        assert_eq!(shape.len(), 5);
        assert_eq!(shape[0], 1); // batch
        assert_eq!(shape[1], 17); // num_crops + 1
        assert_eq!(shape[2], 3); // channels
        assert_eq!(shape[3], 336); // height
        assert_eq!(shape[4], 336); // width

        // Check model-specific outputs
        assert!(result.model_specific.contains_key("image_sizes"));
        assert!(result.model_specific.contains_key("num_img_tokens"));
    }

    #[test]
    fn test_phi3_vision_preprocess_multiple() {
        let processor = Phi3VisionProcessor::new();

        let images = [
            create_test_image(504, 504, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = processor
            .preprocess(&images.iter().collect::<Vec<_>>())
            .unwrap();

        assert_eq!(result.batch_size(), 2);
        assert_eq!(result.item_sizes.len(), 2);
        assert_eq!(result.feature_token_counts.len(), 2);
    }

    #[test]
    fn test_model_name() {
        let processor = Phi3VisionProcessor::new();
        assert_eq!(processor.model_name(), "phi3-vision");
    }

    #[test]
    fn test_from_config() {
        let config = PreProcessorConfig {
            num_crops: Some(8),
            image_mean: Some(vec![0.5, 0.5, 0.5]),
            image_std: Some(vec![0.5, 0.5, 0.5]),
            ..Default::default()
        };

        let processor = Phi3VisionProcessor::from_preprocessor_config(&config);
        assert_eq!(processor.num_crops(), 8);
    }

    #[test]
    fn test_transpose_equivalence() {
        // Test that fliph().rotate270() correctly implements PIL's Image.TRANSPOSE
        // TRANSPOSE swaps x and y coordinates: pixel at (x, y) goes to (y, x)
        use image::{GenericImageView, Rgb, RgbImage};

        let mut img = RgbImage::new(100, 200);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // Top-left = red
        img.put_pixel(99, 0, Rgb([0, 255, 0])); // Top-right = green
        img.put_pixel(0, 199, Rgb([0, 0, 255])); // Bottom-left = blue
        img.put_pixel(99, 199, Rgb([255, 255, 0])); // Bottom-right = yellow

        let img = DynamicImage::ImageRgb8(img);
        let transposed = img.fliph().rotate270();

        // After TRANSPOSE: (x, y) -> (y, x)
        assert_eq!(transposed.get_pixel(0, 0).0[0..3], [255, 0, 0]); // (0,0) -> (0,0)
        assert_eq!(transposed.get_pixel(0, 99).0[0..3], [0, 255, 0]); // (99,0) -> (0,99)
        assert_eq!(transposed.get_pixel(199, 0).0[0..3], [0, 0, 255]); // (0,199) -> (199,0)
        assert_eq!(transposed.get_pixel(199, 99).0[0..3], [255, 255, 0]); // (99,199) -> (199,99)
    }

    /// The pixel values of the checkpoint's own image processor (num_crops 4),
    /// bit for bit in `f32` and on the bfloat16 wire, on four generator
    /// images: a landscape and a portrait (the transposed path) that pad to
    /// 672x672, a 7x5 image, and a 1000x60 one whose HD size is 1344x336 (the
    /// width axis at four times the tile). The digests are SHA-256 of the
    /// processor's `pixel_values` bytes.
    #[test]
    fn pixel_values_match_the_checkpoint_processor() {
        use sha2::{Digest, Sha256};

        use crate::{
            encoder_inputs::EncoderDtype,
            test_support::{hex, lcg_rgb_image},
        };

        let processor = Phi3VisionProcessor::with_config(4);
        for (width, height, seed, size, tokens, f32_digest, bf16_digest) in [
            (
                100,
                70,
                5,
                (672, 672),
                757,
                "4ba40c76a86e532377e6f0d06588ec56b28368f787bd553cd76be9d3f978dba6",
                "1445925777c30b43f0684447a69ba5ce1843dd10a8e243d0d28a05da5a1f04b4",
            ),
            (
                50,
                90,
                6,
                (672, 672),
                757,
                "cc18aa698c7ef428146a204a673ea2718deb2f868fd806ebff1cd06038f70839",
                "d18defa6d67345620762d932ea558315425ba0b260fdc3ac3b9e925c16c5f507",
            ),
            (
                7,
                5,
                7,
                (672, 672),
                757,
                "70d7d5c83cb601d5e6bc462e4fba7985600a7bb0d816ff8c5b60043eead595a1",
                "a2b93fca8a9e60bd909adac8ba72b8a93428b0e7184bab8feabd6bfff6d3e85a",
            ),
            (
                1000,
                60,
                8,
                (336, 1344),
                745,
                "6323dbef746a9113b0d47154eca9c81e7f958dd31f50f9bf2440ce153eac5899",
                "8b206fbd9154dd3708c94f77db1857110d1a9cf92e1491ec722ac97c0182bb53",
            ),
        ] {
            let image = lcg_rgb_image(width, height, seed);
            let result = processor.preprocess(&[&image]).unwrap();
            assert_eq!(
                result.encoder_input.shape(),
                &[1, 5, 3, 336, 336],
                "{width}x{height}"
            );
            assert_eq!(
                result.item_sizes,
                vec![(size.1, size.0)],
                "{width}x{height}"
            );
            assert_eq!(
                result.feature_token_counts,
                vec![tokens],
                "{width}x{height}"
            );
            let digest = |dtype| hex(&Sha256::digest(result.encoder_input.write_as(dtype)));
            assert_eq!(
                digest(EncoderDtype::Float32),
                f32_digest,
                "{width}x{height} f32"
            );
            assert_eq!(
                digest(EncoderDtype::BFloat16),
                bf16_digest,
                "{width}x{height} bf16"
            );
        }
    }

    /// An image with no pixel on a side never reaches the HD transform's scale
    /// loop, which divides by the shorter side and would not end: the batch
    /// and the token count refuse it first, in either orientation and at 0x0.
    #[test]
    fn an_image_with_a_zero_side_is_refused_before_the_hd_transform() {
        let processor = Phi3VisionProcessor::with_config(4);
        for (width, height) in [(0, 7), (7, 0), (0, 0)] {
            let image = DynamicImage::new_rgb8(width, height);
            let error = processor
                .preprocess(&[&image])
                .expect_err("a side with no pixel");
            assert!(
                matches!(&error, TransformError::ShapeError(message)
                    if message.contains(&format!("{width}x{height}"))
                        && message.contains("a side with no pixel")),
                "{width}x{height}: {error}"
            );
            let error = VisionPreProcessor::calculate_num_tokens(&processor, width, height)
                .expect_err("a side with no pixel");
            assert!(
                matches!(&error, TransformError::ShapeError(message)
                    if message.contains("a side with no pixel")),
                "{width}x{height}: {error}"
            );
        }
    }

    /// The token count from the declared size alone agrees with the count the
    /// transformed image gives, at both crop budgets, in both orientations,
    /// whether or not the target's height needs padding to a whole tile.
    #[test]
    fn the_token_count_from_the_declared_size_matches_the_transformed_image() {
        for num_crops in [4, 16] {
            let processor = Phi3VisionProcessor::with_config(num_crops);
            for (width, height) in [
                (336, 336),
                (1344, 1344),
                (1008, 1344),
                (1344, 1008),
                (640, 480),
                (480, 640),
                (1920, 1080),
                (1080, 1920),
                (50, 30),
                (30, 50),
                (1344, 1),
                (1, 1344),
                (4000, 300),
                (300, 4000),
            ] {
                let image = DynamicImage::new_rgb8(width, height);
                let batch = processor.preprocess(&[&image]).expect("served");
                assert_eq!(
                    VisionPreProcessor::calculate_num_tokens(&processor, width, height)
                        .expect("served"),
                    batch.feature_token_counts[0],
                    "{num_crops} crops, {width}x{height}"
                );
            }
        }
    }

    /// An image whose longer side is more than `num_crops * 336` times its
    /// shorter side scales to a target with no pixel on its shorter side
    /// (6000x1 -> 1344x0 at four crops, 1x6000 -> 0x1344), on which the
    /// reference's resize raises: the batch and the token count refuse it
    /// with the sizes named in the image's own orientation, while the 1344x1
    /// edge that keeps a pixel is served.
    #[test]
    fn an_image_with_no_pixel_of_height_after_the_hd_transform_is_refused() {
        let processor = Phi3VisionProcessor::with_config(4);
        for (width, height, target) in [
            (6000, 1, "1344x0"),
            (1, 6000, "0x1344"),
            (2689, 2, "1344x0"),
        ] {
            let image = DynamicImage::new_rgb8(width, height);
            let error = processor
                .preprocess(&[&image])
                .expect_err("no pixel on the shorter side");
            assert!(
                matches!(&error, TransformError::ShapeError(message)
                    if message.contains(target) && message.contains("shorter side")),
                "{width}x{height}: {error}"
            );
            assert!(
                VisionPreProcessor::calculate_num_tokens(&processor, width, height).is_err(),
                "{width}x{height}"
            );
        }
        for (width, height) in [(1344, 1), (1, 1344)] {
            let image = DynamicImage::new_rgb8(width, height);
            assert!(processor.preprocess(&[&image]).is_ok(), "{width}x{height}");
            assert!(
                VisionPreProcessor::calculate_num_tokens(&processor, width, height).is_ok(),
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn an_empty_batch_is_refused_as_empty_batch() {
        let processor = Phi3VisionProcessor::new();
        let result = processor.preprocess(&[]);
        assert!(matches!(result, Err(TransformError::EmptyBatch)));
    }
}
