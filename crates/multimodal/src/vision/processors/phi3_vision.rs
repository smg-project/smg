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

use std::borrow::Cow;

use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{Array3, IxDyn};

use crate::{
    buffer_pool,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
        scratch,
        transforms::{self, TransformError},
    },
};

/// CLIP normalization mean values.
pub(crate) const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization std values.
pub(crate) const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Default number of crops for HD transform.
pub(crate) const DEFAULT_NUM_CROPS: usize = 16;

/// Tile size used in Phi3-Vision (336x336).
pub(crate) const TILE_SIZE: u32 = 336;

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
    pub(crate) fn new() -> Self {
        Self {
            num_crops: DEFAULT_NUM_CROPS,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
    }

    /// Create a processor with custom settings.
    #[cfg(test)]
    fn with_config(num_crops: usize) -> Self {
        Self {
            num_crops,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
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
    pub(crate) fn hd_transform(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();

        let transposed = width < height;
        let img = if transposed {
            // Transpose (PIL's Image.TRANSPOSE): the pixel at (x, y) goes to
            // (y, x). An RGB8 image in one pass, any other through the flip and
            // the quarter turn that make the same image.
            Cow::Owned(match image {
                DynamicImage::ImageRgb8(rgb) => {
                    DynamicImage::ImageRgb8(transforms::transpose_rgb(rgb))
                }
                other => other.fliph().rotate270(),
            })
        } else {
            Cow::Borrowed(image)
        };

        let (width, height) = img.dimensions();
        let (new_w, new_h) = self.hd_target(width, height);

        // The reference resizes the PIL image through torchvision's `resize`
        // at its default interpolation, which is `PIL.Image.resize(BILINEAR)`:
        // Pillow's triangle filter widened by the downscale factor, in its
        // fixed point, two passes.
        let resized = transforms::resize_bilinear_pil(&img, new_w, new_h);

        // Pad height to multiple of 336
        let padded = self.padding_336(resized.into_rgb8());

        // Transpose back if needed (transpose is self-inverse). The image has
        // then been read for the last time and its buffer is offered to the
        // pool. A padded canvas came from the pool; an image that needed no
        // padding still has the resize's buffer, which the pool frees unless
        // it is of one of its classes.
        DynamicImage::ImageRgb8(if transposed {
            let turned = transforms::transpose_rgb(&padded);
            buffer_pool::recycle(padded.into_raw());
            turned
        } else {
            padded
        })
    }

    /// Pad image height to multiple of 336 (centered, white padding). An image
    /// already whole tiles high comes back as given. A padded one has its
    /// pixels in a buffer from the pool's `take` (a plain allocation outside
    /// the pooled range), which its last reader offers back to the pool:
    /// `process_single_image`, or the transpose back.
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the processor; keeps API consistent"
    )]
    fn padding_336(&self, image: RgbImage) -> RgbImage {
        let (width, height) = image.dimensions();
        let target_h = ((height as f64 / TILE_SIZE as f64).ceil() * TILE_SIZE as f64) as u32;

        if height == target_h {
            return image;
        }

        let top_padding = (target_h - height) / 2;

        // The canvas is a buffer of the pool, so its bytes are a previous
        // request's and every one of them is written here: the image's rows
        // are one block in its middle (the two have one width), and the band
        // above them and the band below are white.
        let rows = image.as_raw();
        let top = top_padding as usize * width as usize * 3;
        let mut canvas = buffer_pool::take(width as usize * target_h as usize * 3);
        let (above, rest) = canvas.split_at_mut(top);
        let (middle, below) = rest.split_at_mut(rows.len());
        above.fill(255);
        middle.copy_from_slice(rows);
        below.fill(255);
        #[expect(
            clippy::expect_used,
            reason = "the canvas holds exactly width * target_h * 3 bytes by construction"
        )]
        RgbImage::from_raw(width, target_h, canvas).expect("padded canvas size")
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

    /// Process a single image through the full pipeline and append its
    /// `[num_crops + 1, 3, 336, 336]` values to `values`: the global image,
    /// the HD image's tiles in row-major order, zeros for the crops the image
    /// does not fill.
    fn process_single_image(
        &self,
        image: &DynamicImage,
        values: &mut Vec<f32>,
    ) -> ((usize, usize), usize) {
        // 1. Convert to RGB
        let image = match image {
            DynamicImage::ImageRgb8(_) => Cow::Borrowed(image),
            other => Cow::Owned(DynamicImage::ImageRgb8(other.to_rgb8())),
        };

        // 2. HD transform
        let hd_image = self.hd_transform(&image);
        let (hd_w, hd_h) = hd_image.dimensions();

        // 3. To tensor [0, 1] and normalize, in torchvision's steps (the
        //    reference composes `ToTensor` and `Normalize`: three roundings)
        let tensor = transforms::to_tensor_then_normalize(&hd_image, &self.mean, &self.std);
        // The HD image has been read for the last time: its buffer is offered
        // to the pool, which keeps a padding's canvas for the next request
        // (`padding_336` says what the buffer of an unpadded image is).
        buffer_pool::recycle_image(hd_image);

        // 4. The global image (336x336) comes first
        let tile = TILE_SIZE as usize;
        let end = values.len() + (self.num_crops + 1) * 3 * tile * tile;
        let (global, _) = self.create_global_image(&tensor).into_raw_vec_and_offset();
        values.extend_from_slice(&global);

        // 5. The HD image's tiles, row-major, as far as the crops go
        let (h, w) = (hd_h as usize, hd_w as usize);
        let (planes, _) = tensor.into_raw_vec_and_offset();
        'tiles: for grid_y in 0..h / tile {
            for grid_x in 0..w / tile {
                if values.len() + 3 * tile * tile > end {
                    break 'tiles;
                }
                for plane in planes.chunks_exact(h * w) {
                    for row in plane.chunks_exact(w).skip(grid_y * tile).take(tile) {
                        values.extend_from_slice(&row[grid_x * tile..(grid_x + 1) * tile]);
                    }
                }
            }
        }
        scratch::give_f32(planes);

        // 6. Zeros for the crops the image does not fill
        values.resize(end, 0.0);

        // Calculate token count
        let num_tokens = self.calculate_num_tokens(h, w);

        // The returned size is the HD-transformed image size.
        ((h, w), num_tokens)
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
        TransformError::if_empty(images)?;

        for image in images {
            let (width, height) = image.dimensions();
            self.hd_target_or_refuse(width, height)?;
        }

        // The batch [B, num_crops+1, 3, 336, 336], written once per image.
        // Pooled: it is the encoder input's storage, which returns to the
        // pool when the input is written out.
        let max_crops = self.num_crops + 1;
        let batch_size = images.len();
        let tile = TILE_SIZE as usize;
        let shape = vec![batch_size, max_crops, 3, tile, tile];
        let refused = |reason: String| format!("cannot reserve Phi-3 patch output: {reason}");
        let len = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| {
                TransformError::ReserveRefused(refused(format!("{shape:?} overflows usize")))
            })?;
        let mut flat_data = scratch::try_take_f32_cap(len)
            .map_err(|error| TransformError::ReserveRefused(refused(error.to_string())))?;
        let mut all_image_sizes = Vec::with_capacity(images.len());
        let mut all_num_tokens = Vec::with_capacity(images.len());

        for image in images {
            let (image_size, num_tokens) = self.process_single_image(image, &mut flat_data);
            all_image_sizes.push((image_size.1 as u32, image_size.0 as u32)); // (width, height)
            all_num_tokens.push(num_tokens);
        }

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
    use image::Rgb;

    use super::*;
    use crate::test_support::{create_test_image, lcg_bytes};

    #[test]
    fn test_phi3_vision_processor_default() {
        let processor = Phi3VisionProcessor::new();
        assert_eq!(processor.num_crops, 16);
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
        assert!(num_tiles <= processor.num_crops as u32);
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

    /// A batch is its images' own outputs in turn, and the crops an image
    /// does not fill are zeros: an image's slot starts where the slot before
    /// it ends, whatever that image filled.
    #[test]
    fn a_batch_is_its_images_in_turn_and_unfilled_crops_are_zero() {
        use crate::test_support::lcg_rgb_image;

        let processor = Phi3VisionProcessor::new();
        let images = [
            lcg_rgb_image(300, 200, 1),
            lcg_rgb_image(90, 700, 2),
            lcg_rgb_image(700, 400, 3),
        ];
        let crop = 3 * (TILE_SIZE * TILE_SIZE) as usize;
        let mut joined = Vec::new();
        let mut filled_crops = Vec::new();
        for image in &images {
            let single = processor.preprocess(&[image]).unwrap();
            let (width, height) = single.item_sizes[0];
            let filled = 1 + ((width / TILE_SIZE) * (height / TILE_SIZE)) as usize;
            let values = single.encoder_input.as_f32();
            let values = values.as_slice().unwrap();
            assert!(values[filled * crop..]
                .iter()
                .all(|value| value.to_bits() == 0));
            joined.extend_from_slice(values);
            filled_crops.push(filled);
        }
        // Two of the three leave crops unfilled; the tall one fills them all.
        assert_eq!(filled_crops, [13, 17, 16]);

        let batch = processor
            .preprocess(&images.iter().collect::<Vec<_>>())
            .unwrap();
        assert_eq!(batch.encoder_input.as_f32().as_slice().unwrap(), joined);
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
        assert_eq!(processor.num_crops, 8);
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

    /// A batch of 2^40 crops is more bytes than any allocation holds: the
    /// request reads an error.
    #[test]
    fn a_batch_that_cannot_be_reserved_is_an_error() {
        let processor = Phi3VisionProcessor::with_config(1 << 40);
        let image = create_test_image(1, 1, Rgb([0, 0, 0]));

        let error = processor.preprocess(&[&image]).expect_err("not reserved");

        assert!(
            matches!(&error, TransformError::ReserveRefused(message)
                if message.contains("cannot reserve Phi-3 patch output")),
            "{error}"
        );
    }

    /// A batch of 2^46 crops is more elements than `usize` counts: the size
    /// overflows before any room is asked, and the request reads an error.
    #[test]
    fn a_batch_whose_size_overflows_is_an_error() {
        let processor = Phi3VisionProcessor::with_config(1 << 46);
        let image = create_test_image(1, 1, Rgb([0, 0, 0]));

        let error = processor.preprocess(&[&image]).expect_err("overflows");

        assert!(
            matches!(&error, TransformError::ReserveRefused(message)
                if message.contains("overflows usize")),
            "{error}"
        );
    }

    /// The padded canvas is a buffer of the pool. Its bytes are a previous
    /// request's (the sentinel in a debug build), so the image's rows and
    /// the two white bands must cover every one of them; and a request
    /// returns the buffer, for a wide image (the canvas is the HD image) and
    /// for a tall one (the canvas is transposed back, then returned).
    #[test]
    fn the_padded_canvas_is_taken_from_the_pool_and_returned_to_it() {
        static POOL: buffer_pool::BufferPool = buffer_pool::BufferPool::new();
        buffer_pool::with_pool(&POOL, || {
            let processor = Phi3VisionProcessor::with_config(4);
            // 672x378 pads to 672x672: 147 white rows above and 147 below.
            let rows = RgbImage::from_raw(672, 378, lcg_bytes(672 * 378 * 3, 7)).expect("672x378");
            let mut expected = RgbImage::from_pixel(672, 672, Rgb([255, 255, 255]));
            image::imageops::replace(&mut expected, &rows, 0, 147);
            let class = buffer_pool::allocation_capacity(expected.as_raw().len());
            for round in 0..2 {
                let canvas = processor.padding_336(rows.clone());
                assert!(canvas == expected, "round {round}: pixels differ");
                assert_eq!(
                    POOL.held(),
                    0,
                    "round {round}: the canvas is out of the pool"
                );
                buffer_pool::recycle(canvas.into_raw());
                assert_eq!(
                    POOL.held(),
                    class,
                    "round {round}: the canvas is idle in the pool"
                );
            }
            for (width, height) in [(1920, 1080), (1080, 1920)] {
                let image = create_test_image(width, height, Rgb([90, 120, 150]));
                processor.preprocess(&[&image]).expect("a photo");
                assert_eq!(
                    POOL.held(),
                    class,
                    "{width}x{height}: the request returned the canvas"
                );
            }
        });
    }
}
