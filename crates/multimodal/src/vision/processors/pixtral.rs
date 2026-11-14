//! Pixtral/Mistral3 Vision image processor implementation.
//!
//! This module implements the image preprocessing for Pixtral/Mistral3 models,
//! matching the behavior of HuggingFace's `PixtralImageProcessor`.
//!
//! Key characteristics:
//! - CLIP normalization: mean [0.48145466, 0.4578275, 0.40821073], std [0.26862954, 0.26130258, 0.27577711]
//! - Bicubic resampling for resize: the torchvision `uint8` antialiased kernel
//!   the reference runs ([`resize_bicubic_torch_aa_rgb`])
//! - Images resized to fit within longest_edge (default 1024)
//! - Output dimensions are multiples of patch_size (default 16)
//! - No tiling - single image output per input
//! - The output is the image's bytes with their normalization
//!   ([`EncoderInput::U8`]): a float destination takes, per byte, the value
//!   of the reference's fused rescale-and-normalize step. A batch of images
//!   of different sizes is the one output in floats ([`EncoderInput::F32`]):
//!   the reference pads such a batch to its largest image after normalizing,
//!   with 0.0, a value no byte maps to

use std::borrow::Cow;

use image::DynamicImage;
use ndarray::Array4;

use crate::{
    encoder_inputs::{EncoderInput, ModelSpecificValue, PixelNorm, PreprocessedEncoderInputs},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::VisionPreProcessor,
        processors::qwen_vl_base::pixel_norm,
        scratch,
        transforms::{resize_bicubic_torch_aa_rgb, rgb_bytes, TransformError},
    },
};

/// Default normalization mean values (CLIP)
const DEFAULT_IMAGE_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// Default normalization std values (CLIP)
const DEFAULT_IMAGE_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Default longest edge for resize
const DEFAULT_LONGEST_EDGE: u32 = 1024;

/// Default patch size
const DEFAULT_PATCH_SIZE: u32 = 16;

/// Pixtral/Mistral3 Vision image processor.
///
/// This processor handles image preprocessing for Pixtral and Mistral3 vision models.
/// Unlike tile-based processors (Phi3, LLaMA4), Pixtral processes images at their
/// natural resolution (up to a maximum), preserving aspect ratio.
#[derive(Debug, Clone)]
pub struct PixtralProcessor {
    /// Maximum dimension for the longest edge
    longest_edge: u32,
    /// Patch size for calculating output dimensions
    patch_size: u32,
    /// Normalization mean values
    image_mean: [f64; 3],
    /// Normalization std values
    image_std: [f64; 3],
    /// The byte-to-float map of the config's rescale and normalize steps, in
    /// the reference's fused arithmetic; its channel block is one image plane
    /// of the batch, set per call.
    norm: PixelNorm,
}

impl Default for PixtralProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl PixtralProcessor {
    /// Creates a new Pixtral processor with default settings.
    pub fn new() -> Self {
        Self::from_preprocessor_config(&PreProcessorConfig::default())
    }

    /// Creates a processor from a HuggingFace preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let longest_edge = config
            .size
            .as_ref()
            .and_then(|s| s.get("longest_edge").copied())
            .unwrap_or(DEFAULT_LONGEST_EDGE);

        // Patch size uses the new PatchSize type from config
        let patch_size = config.get_patch_size(DEFAULT_PATCH_SIZE as usize) as u32;

        let image_mean = config
            .image_mean
            .as_ref()
            .filter(|m| m.len() >= 3)
            .map(|m| [m[0], m[1], m[2]])
            .unwrap_or(DEFAULT_IMAGE_MEAN);

        let image_std = config
            .image_std
            .as_ref()
            .filter(|s| s.len() >= 3)
            .map(|s| [s[0], s[1], s[2]])
            .unwrap_or(DEFAULT_IMAGE_STD);

        Self {
            longest_edge,
            patch_size,
            image_mean,
            image_std,
            norm: pixel_norm(config, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD, 1),
        }
    }

    /// Calculates the target output size for an image.
    ///
    /// The image is resized to fit within `longest_edge` while preserving aspect ratio.
    /// The output dimensions are then adjusted to be multiples of `patch_size`.
    fn get_resize_output_size(&self, height: u32, width: u32) -> (u32, u32) {
        let max_size = self.longest_edge;
        let patch_size = self.patch_size;

        // Calculate ratio for scaling down (only if larger than max_size)
        let ratio = f64::max(
            height as f64 / max_size as f64,
            width as f64 / max_size as f64,
        );

        let (new_height, new_width) = if ratio > 1.0 {
            // Scale down using floor to ensure we don't exceed max_size
            let new_height = (height as f64 / ratio).floor() as u32;
            let new_width = (width as f64 / ratio).floor() as u32;
            (new_height, new_width)
        } else {
            (height, width)
        };

        // Calculate number of patches in each dimension
        // Using: num_tokens = (dim - 1) / patch_size + 1 (i.e., ceiling division)
        let num_height_tokens = (new_height.max(1) - 1) / patch_size + 1;
        let num_width_tokens = (new_width.max(1) - 1) / patch_size + 1;

        // Final size is patches * patch_size
        (
            num_height_tokens * patch_size,
            num_width_tokens * patch_size,
        )
    }

    /// One image resized to its target size, as its R, G and B planes back to
    /// back (`3 * height * width` bytes), with that `(height, width)`.
    fn planes_of(&self, image: &DynamicImage) -> Result<(Vec<u8>, (usize, usize)), TransformError> {
        let (width, height, rgb) = rgb_bytes(image);
        let (target_h, target_w) = self.get_resize_output_size(height as u32, width as u32);
        // An image already at its target size passes through, as the
        // reference's resize does.
        let resized = if (width, height) == (target_w as usize, target_h as usize) {
            rgb
        } else {
            Cow::Owned(
                resize_bicubic_torch_aa_rgb(&rgb, width as u32, height as u32, target_w, target_h)?
                    .into_raw(),
            )
        };
        let pixels = target_h as usize * target_w as usize;
        let mut planes = vec![0u8; 3 * pixels];
        let (r_plane, rest) = planes.split_at_mut(pixels);
        let (g_plane, b_plane) = rest.split_at_mut(pixels);
        for (((rgb, r), g), b) in resized
            .as_chunks::<3>()
            .0
            .iter()
            .zip(r_plane.iter_mut())
            .zip(g_plane.iter_mut())
            .zip(b_plane.iter_mut())
        {
            *r = rgb[0];
            *g = rgb[1];
            *b = rgb[2];
        }
        Ok((planes, (target_h as usize, target_w as usize)))
    }
}

impl VisionPreProcessor for PixtralProcessor {
    /// Bytes for every single image and same-size batch, the forms a
    /// destination that normalizes on device takes. A batch of images of
    /// different sizes is the exception: the reference pads it after
    /// normalizing, with 0.0, which no byte expresses, so that batch is
    /// floats ([`EncoderInput::F32`]) and such a destination has no bytes to
    /// take for it.
    fn emits_pixel_bytes(&self) -> bool {
        true
    }

    fn default_mean(&self) -> [f64; 3] {
        self.image_mean
    }

    fn default_std(&self) -> [f64; 3] {
        self.image_std
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let mut planes = Vec::with_capacity(images.len());
        let mut original_sizes = Vec::with_capacity(images.len());
        let mut feature_token_counts = Vec::with_capacity(images.len());
        for image in images {
            planes.push(self.planes_of(image)?);
            original_sizes.push((image.height(), image.width()));
            feature_token_counts.push(self.calculate_num_tokens(image.width(), image.height())?);
        }

        let max_height = planes.iter().map(|(_, (h, _))| *h).max().unwrap_or(0);
        let max_width = planes.iter().map(|(_, (_, w))| *w).max().unwrap_or(0);
        let plane = max_height * max_width;
        let batch_size = planes.len();
        let shape = (batch_size, 3, max_height, max_width);
        let norm = PixelNorm {
            channel_block: plane,
            ..self.norm
        };
        let encoder_input = if planes
            .iter()
            .all(|(_, size)| *size == (max_height, max_width))
        {
            // Every image at the batch's size: the bytes, a single image's
            // moved rather than copied.
            let batch = if let [(single, _)] = planes.as_mut_slice() {
                std::mem::take(single)
            } else {
                let mut batch = Vec::with_capacity(batch_size * 3 * plane);
                for (image_planes, _) in &planes {
                    batch.extend_from_slice(image_planes);
                }
                batch
            };
            EncoderInput::bytes(
                Array4::from_shape_vec(shape, batch)
                    .map_err(|e| TransformError::ShapeError(e.to_string()))?,
                norm,
            )
        } else {
            // Images of different sizes: the reference pads the batch to its
            // largest image (top-left aligned) after normalizing, so the
            // padding is 0.0, a value no byte maps to. The batch is the
            // floats: every byte through the table a float destination would
            // apply, 0.0 beyond each image's own grid.
            let table = norm.table_f32();
            let mut batch = scratch::take_f32(batch_size * 3 * plane);
            for ((image_planes, (height, width)), image) in
                planes.iter().zip(batch.chunks_exact_mut(3 * plane))
            {
                let channels = image_planes
                    .chunks_exact(height * width)
                    .zip(image.chunks_exact_mut(plane));
                for ((source, canvas), map) in channels.zip(&table) {
                    for (row, canvas_row) in source
                        .chunks_exact(*width)
                        .zip(canvas.chunks_exact_mut(max_width))
                    {
                        for (value, &byte) in canvas_row.iter_mut().zip(row) {
                            *value = map[usize::from(byte)];
                        }
                    }
                }
            }
            EncoderInput::from(
                Array4::from_shape_vec(shape, batch)
                    .map_err(|e| TransformError::ShapeError(e.to_string()))?,
            )
        };

        let image_sizes = planes
            .iter()
            .flat_map(|(_, (h, w))| [*h as i64, *w as i64])
            .collect();
        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, original_sizes)
                .with_extra(
                    "image_sizes",
                    ModelSpecificValue::int_2d(image_sizes, batch_size, 2),
                ),
        )
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let (target_h, target_w) = self.get_resize_output_size(height, width);
        let patch_size = self.patch_size;

        // Number of tokens = num_patches_h * num_patches_w
        let num_patches_h = target_h / patch_size;
        let num_patches_w = target_w / patch_size;
        Ok((num_patches_h * num_patches_w) as usize)
    }

    fn model_name(&self) -> &'static str {
        "pixtral"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        // Pixtral has dynamic size based on input
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use image::{Rgb, RgbImage};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::{
        encoder_inputs::EncoderDtype,
        test_support::{hex, lcg_rgb_image},
        vision::preprocessor_config::PatchSize,
    };

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = ((x * 255) / width.max(1)) as u8;
                let g = ((y * 255) / height.max(1)) as u8;
                let b = (((x + y) * 128) / (width + height).max(1)) as u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    /// The checkpoint's preprocessor config (longest edge 1024, 16-pixel
    /// patches, CLIP mean and std, rescale by 1/255).
    fn checkpoint_config() -> PreProcessorConfig {
        PreProcessorConfig::from_json(
            r#"{"image_processor_type":"PixtralImageProcessor","size":{"longest_edge":1024},
                "patch_size":{"height":16,"width":16},"resample":3,
                "image_mean":[0.48145466,0.4578275,0.40821073],
                "image_std":[0.26862954,0.26130258,0.27577711],
                "do_resize":true,"do_rescale":true,"rescale_factor":0.00392156862745098,
                "do_normalize":true,"do_convert_rgb":true}"#,
        )
        .unwrap()
    }

    #[test]
    fn test_resize_output_size_small_image() {
        let processor = PixtralProcessor::new();

        // Small image that doesn't need resizing - just pad to patch boundary
        // 100x100 -> patches: ceil(100/16) = 7, output: 7*16 = 112
        let (h, w) = processor.get_resize_output_size(100, 100);
        assert_eq!((h, w), (112, 112));
    }

    #[test]
    fn test_resize_output_size_large_image() {
        let processor = PixtralProcessor::new();

        // Large image that needs resizing
        // 2048x1024: ratio = 2048/1024 = 2.0
        // scaled: 2048/2 = 1024, 1024/2 = 512
        // patches h: ceil(1024/16) = 64, patches w: ceil(512/16) = 32
        // output: 64*16 = 1024, 32*16 = 512
        let (h, w) = processor.get_resize_output_size(2048, 1024);
        assert_eq!((h, w), (1024, 512));
    }

    #[test]
    fn test_resize_output_size_at_limit() {
        let processor = PixtralProcessor::new();

        // Image exactly at limit
        // 1024x768: ratio = max(1024/1024, 768/1024) = 1.0
        // No resize needed
        // patches h: ceil(1024/16) = 64, patches w: ceil(768/16) = 48
        // output: 64*16 = 1024, 48*16 = 768
        let (h, w) = processor.get_resize_output_size(1024, 768);
        assert_eq!((h, w), (1024, 768));
    }

    #[test]
    fn test_process_single_image() {
        let processor = PixtralProcessor::new();
        let image = create_test_image(200, 150);

        let (planes, size) = processor.planes_of(&image).unwrap();

        // 200x150 -> patches h: ceil(150/16) = 10, patches w: ceil(200/16) = 13
        // output: 10*16 = 160, 13*16 = 208
        assert_eq!(size, (160, 208));
        assert_eq!(planes.len(), 3 * 160 * 208);
    }

    /// Two images of different sizes: the batch is padded to the larger
    /// grid, and the padding is the 0.0 the reference writes after
    /// normalizing, so the batch is floats; inside its own grid every image
    /// holds the values its bytes map to (the same values its single-image
    /// byte form takes).
    #[test]
    fn a_mixed_size_batch_is_floats_padded_with_the_references_zero() {
        let processor = PixtralProcessor::new();

        let images = [create_test_image(200, 150), create_test_image(300, 100)];

        let result = processor
            .preprocess(&images.iter().collect::<Vec<_>>())
            .unwrap();

        // First image: 150x200 -> 160x208
        // Second image: 100x300 -> 112x304 (ceil(100/16)=7, ceil(300/16)=19)
        // Batch padded to max: 160x304
        assert_eq!(result.encoder_input.shape(), &[2, 3, 160, 304]);
        assert_eq!(result.feature_token_counts, vec![130, 133]);
        let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("image_sizes")
        else {
            panic!("image_sizes expected");
        };
        assert_eq!(data, &[160, 208, 112, 304]);
        assert_eq!(shape, &[2, 2]);
        let EncoderInput::F32(values) = &result.encoder_input else {
            panic!("a mixed-size batch is floats");
        };
        // The padding beyond each image's grid is +0.0 in every element, as the
        // reference pads after normalizing; a zero byte would map elsewhere.
        assert!(values
            .slice(ndarray::s![0, .., .., 208..])
            .iter()
            .all(|value| value.to_bits() == 0));
        assert!(values
            .slice(ndarray::s![1, .., 112.., ..])
            .iter()
            .all(|value| value.to_bits() == 0));
        assert!(processor.norm.table_f32()[0][0] != 0.0);
        // Inside its grid an image holds the values of its own single-image
        // byte form, bit for bit, in f32 and on the bfloat16 wire.
        for (i, (image, (h, w))) in images.iter().zip([(160, 208), (112, 304)]).enumerate() {
            let single = processor.preprocess(&[image]).unwrap();
            assert!(matches!(single.encoder_input, EncoderInput::U8 { .. }));
            let single_values = single.encoder_input.as_f32().into_owned();
            assert_eq!(
                values.slice(ndarray::s![i, .., ..h, ..w]),
                single_values.slice(ndarray::s![0, .., .., ..])
            );
        }
        let wire = result.encoder_input.write_as(EncoderDtype::BFloat16);
        let (rows, _) = wire.as_chunks::<2>();
        let padded = |element: usize| {
            let (x, y) = (element % 304, element / 304 % 160);
            let image = element / (3 * 160 * 304);
            (image == 0 && x >= 208) || (image == 1 && y >= 112)
        };
        assert!(rows
            .iter()
            .enumerate()
            .filter(|(element, _)| padded(*element))
            .all(|(_, bytes)| *bytes == [0, 0]));
    }

    /// Images of one size pad nothing and stay bytes, each image's planes
    /// back to back in the batch.
    #[test]
    fn a_same_size_batch_stays_bytes() {
        let processor = PixtralProcessor::new();
        let images = [create_test_image(200, 150), lcg_rgb_image(200, 150, 9)];

        let result = processor
            .preprocess(&images.iter().collect::<Vec<_>>())
            .unwrap();

        assert_eq!(result.encoder_input.shape(), &[2, 3, 160, 208]);
        let EncoderInput::U8 { patches, norm } = &result.encoder_input else {
            panic!("pixel bytes expected");
        };
        assert_eq!(norm.channel_block, 160 * 208);
        for (i, image) in images.iter().enumerate() {
            let (planes, _) = processor.planes_of(image).unwrap();
            assert_eq!(
                patches
                    .slice(ndarray::s![i, .., .., ..])
                    .iter()
                    .copied()
                    .collect::<Vec<u8>>(),
                planes
            );
        }
    }

    /// The reference's rule on a mixed-size batch of solid images, written
    /// out: inside its grid an image holds the fused normalize value of its
    /// level (the bits the single-image test above pins), beyond it +0.0;
    /// the bfloat16 wire carries the same zeros.
    #[test]
    fn a_mixed_size_batch_of_solid_images_takes_the_references_values() {
        let processor = PixtralProcessor::from_preprocessor_config(&checkpoint_config());
        let small = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb([0; 3])));
        let large = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([255; 3])));
        let result = processor.preprocess(&[&small, &large]).unwrap();
        assert_eq!(result.encoder_input.shape(), &[2, 3, 32, 32]);
        let values = result
            .encoder_input
            .as_f32()
            .into_owned()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();
        let level_0 = [0xbfe5_68dc_u32, 0xbfe0_44b7, 0xbfbd_77d8];
        let level_255 = [0x3ff7_1542_u32, 0x4004_cae4, 0x4009_5661];
        for channel in 0..3 {
            let small_plane = values.index_axis(ndarray::Axis(0), 0);
            let small_plane = small_plane.index_axis(ndarray::Axis(0), channel);
            for ((y, x), value) in small_plane.indexed_iter() {
                let expected = if y < 16 && x < 16 {
                    level_0[channel]
                } else {
                    0
                };
                assert_eq!(
                    value.to_bits(),
                    expected,
                    "small image channel {channel} ({y}, {x})"
                );
            }
            let large_plane = values.index_axis(ndarray::Axis(0), 1);
            assert!(large_plane
                .index_axis(ndarray::Axis(0), channel)
                .iter()
                .all(|value| value.to_bits() == level_255[channel]));
        }
        let wire = result.encoder_input.write_as(EncoderDtype::BFloat16);
        let (elements, _) = wire.as_chunks::<2>();
        let zeros = elements.iter().filter(|bytes| **bytes == [0, 0]).count();
        assert_eq!(zeros, 3 * (32 * 32 - 16 * 16));
    }

    #[test]
    fn test_normalization_values() {
        let processor = PixtralProcessor::new();

        // Verify CLIP normalization values
        assert!((processor.image_mean[0] - 0.48145466).abs() < 1e-6);
        assert!((processor.image_mean[1] - 0.4578275).abs() < 1e-6);
        assert!((processor.image_mean[2] - 0.40821073).abs() < 1e-6);

        assert!((processor.image_std[0] - 0.26862954).abs() < 1e-6);
        assert!((processor.image_std[1] - 0.26130258).abs() < 1e-6);
        assert!((processor.image_std[2] - 0.27577711).abs() < 1e-6);
    }

    #[test]
    fn test_from_config() {
        let mut size = HashMap::new();
        size.insert("longest_edge".to_string(), 2048u32);

        let config = PreProcessorConfig {
            size: Some(size),
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            image_mean: Some(vec![0.5, 0.5, 0.5]),
            image_std: Some(vec![0.5, 0.5, 0.5]),
            ..Default::default()
        };

        let processor = PixtralProcessor::from_preprocessor_config(&config);

        assert_eq!(processor.longest_edge, 2048);
        assert_eq!(processor.patch_size, 14);
        assert_eq!(processor.image_mean, [0.5, 0.5, 0.5]);
        assert_eq!(processor.image_std, [0.5, 0.5, 0.5]);
        // The byte table carries the config's mean and std in byte units.
        assert_eq!(processor.norm.scale, 1.0);
        assert_eq!(processor.norm.offset, [127.5; 3]);
        assert_eq!(processor.norm.divisor, [127.5; 3]);
    }

    #[test]
    fn test_calculate_num_tokens() {
        let processor = PixtralProcessor::new();

        // 200x150 -> 208x160 -> 13*10 = 130 patches
        let tokens = processor.calculate_num_tokens(200, 150).unwrap();
        assert_eq!(tokens, 130);
    }

    #[test]
    fn an_empty_batch_is_refused_as_empty_batch() {
        let processor = PixtralProcessor::new();
        let result = processor.preprocess(&[]);
        assert!(matches!(result, Err(TransformError::EmptyBatch)));
    }

    /// A byte maps to the value the reference's fused rescale-and-normalize
    /// step gives it: `(byte - mean * 255) / (std * 255)` in `f32`, two
    /// roundings, not the three of a rescale followed by a normalize. The
    /// bits are the reference processor's on solid 16x16 images.
    #[test]
    fn a_solid_image_takes_the_fused_normalize_values_of_the_reference() {
        let processor = PixtralProcessor::from_preprocessor_config(&checkpoint_config());
        for (level, bits) in [
            (0u8, [0xbfe5_68dc_u32, 0xbfe0_44b7, 0xbfbd_77d8]),
            (1, [0xbfe3_8a80, 0xbfde_58f1, 0xbfbb_a5e2]),
            (128, [0x3d9c_560e, 0x3e2c_f360, 0x3eae_0dc1]),
            (254, [0x3ff5_36e6, 0x4003_d501, 0x4008_6d66]),
            (255, [0x3ff7_1542, 0x4004_cae4, 0x4009_5661]),
        ] {
            let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb([level; 3])));
            let result = processor.preprocess(&[&image]).unwrap();
            let values = result.encoder_input.as_f32().into_owned();
            assert_eq!(values.shape(), &[1, 3, 16, 16]);
            for (channel, expected) in bits.into_iter().enumerate() {
                let plane = values.index_axis(ndarray::Axis(1), channel);
                assert!(
                    plane.iter().all(|value| value.to_bits() == expected),
                    "level {level} channel {channel}: {:#x} vs {expected:#x}",
                    plane[[0, 0, 0]].to_bits()
                );
            }
        }
    }

    /// The pixel values of the reference processor, bit for bit, in `f32`
    /// and as the bfloat16 wire: an image at its own size (the table alone),
    /// one the grid rounds up (an upscale to the next patch multiple) and one
    /// over the longest edge (a downscale with the antialiased kernel). The
    /// digests are SHA-256 of the reference's `pixel_values` bytes.
    #[test]
    fn pixels_match_the_reference_processor() {
        let processor = PixtralProcessor::from_preprocessor_config(&checkpoint_config());
        for (width, height, seed, shape, f32_digest, bf16_digest) in [
            (
                32,
                32,
                3,
                [1, 3, 32, 32],
                "70880016be918c85c1e585ceb76b771d60855cf811c32d15f1e6f8a943892bd8",
                "7cdf2d3fd07068742abd8290fc08451cba7a97a534b9d8297caa6489d1999cca",
            ),
            (
                40,
                24,
                5,
                [1, 3, 32, 48],
                "5f962d1a6bd51aa9ae791a8c5c41415604aef9e590ca9825d724b5ab43697a6d",
                "a759c0e51c20794b4cee4855232ece722b540bfa8159007d89dc0271cbc33df9",
            ),
            (
                1100,
                50,
                7,
                [1, 3, 48, 1024],
                "61e40e44f3a3ec9341cdd83be6507378fbaa15976e255079856aaaa0f38ad554",
                "fe2efb3ca17a4f269e87df85834b550e9bfc5f96968b24470617a1215c764c4c",
            ),
        ] {
            let image = lcg_rgb_image(width, height, seed);
            let result = processor.preprocess(&[&image]).unwrap();
            assert_eq!(result.encoder_input.shape(), &shape, "{width}x{height}");
            assert_eq!(
                hex(&Sha256::digest(
                    result.encoder_input.write_as(EncoderDtype::Float32)
                )),
                f32_digest,
                "{width}x{height} f32"
            );
            assert_eq!(
                hex(&Sha256::digest(
                    result.encoder_input.write_as(EncoderDtype::BFloat16)
                )),
                bf16_digest,
                "{width}x{height} bf16"
            );
        }
    }
}
