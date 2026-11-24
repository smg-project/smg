//! Inkling2 native still-image preprocessing, without training-only virtual resizing.
//!
//! Features retain separate F32 divide/subtract/divide normalization; a
//! bfloat16 or float16 wire narrows each level of it in the processor's own
//! per-byte table, any other destination in the common wire conversion. Grid
//! cells are pixel-major RGB patches with two identical temporal slices; every
//! cell produces one language-model token.

use std::collections::{HashMap, TryReserveError};

use image::{DynamicImage, GenericImageView};
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Value};

use crate::{
    buffer_pool,
    encoder_inputs::{f32_to_bf16_bits, f32_to_f16_bits, EncoderDtype, EncoderInput, PixelNorm},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::TransformError,
    },
};

const PATCH_SIZE: usize = 40;
const TEMPORAL_PATCH_SIZE: usize = 2;
const PATCH_VALUES: usize = TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE * 3;
// Keep the existing Inkling allocation bound; oversized inputs fail without resizing.
const MAX_PATCHES: usize = super::inkling::DEFAULT_INKLING_MAX_PATCHES;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];

#[derive(Debug, Clone, Copy, Default)]
pub struct Inkling2ImageProcessor;

impl Inkling2ImageProcessor {
    pub fn new() -> Self {
        Self
    }

    /// The processor for `config`, which must be one of the checkpoint's
    /// two forms of the fixed policy (see [`Self::validate_config`]); any
    /// other config is refused here, before a request.
    pub(crate) fn from_preprocessor_config(
        config: &PreProcessorConfig,
    ) -> Result<Self, TransformError> {
        Self::validate_config(config)?;
        Ok(Self)
    }

    /// Require the checkpoint's fixed preprocessing policy, in either form:
    ///
    /// - the native `image_preprocessing.json`;
    /// - the `Inkling2ImageProcessor` section of the released
    ///   `processor_config.json` (Transformers): 40x40 patches, RGB
    ///   conversion, `1/255` rescale and mean/std 0.5. A field it leaves out
    ///   takes the Transformers class default, which is the same value.
    ///
    /// Both forms describe the same images. The processor sends the wire
    /// format the engine expects in either case. Missing fields, training
    /// virtual-size controls, and other normalization or resize settings fail
    /// rather than change the image semantics.
    pub fn validate_config(config: &PreProcessorConfig) -> Result<(), TransformError> {
        if Self::is_native_policy(config) || Self::is_transformers_policy(config) {
            return Ok(());
        }
        Err(TransformError::ShapeError(
            "Inkling2 requires its native image_preprocessing.json or the released processor_config.json Inkling2ImageProcessor section; resize, virtual-size and normalization overrides are unsupported".to_string(),
        ))
    }

    /// Whether a generic field that neither policy form sets is present.
    fn has_generic_override(config: &PreProcessorConfig) -> bool {
        config.do_pad.is_some()
            || config.do_resize.is_some()
            || config.do_center_crop.is_some()
            || config.resampling.is_some()
            || config.crop_size.is_some()
            || config.merge_size.is_some()
            || config.min_pixels.is_some()
            || config.max_pixels.is_some()
            || config.num_crops.is_some()
            || config.dynamic_hd.is_some()
            || config.max_image_tiles.is_some()
            || config.num_img_tokens.is_some()
            || config.im_start_token.is_some()
            || config.im_end_token.is_some()
            || config.slice_start_token.is_some()
            || config.slice_end_token.is_some()
            || config.vision_start_token.is_some()
            || config.vision_end_token.is_some()
    }

    fn is_native_policy(config: &PreProcessorConfig) -> bool {
        let expected = json!({
            "include_extra_column": false,
            "rescale_image_frac": null,
            "rescale_image_max_upscaled_long_edge": null,
            "stretch_to_padded_grid": false,
            "overlay_rgba_images_on_checkerboard": true,
            "requires_image_grid_thw": true
        });
        let native_extra = Value::Object(config.extra.clone().into_iter().collect());
        config
            .patch_size
            .as_ref()
            .is_some_and(|size| size.height == Some(40) && size.width == Some(40))
            && config.temporal_patch_size == Some(TEMPORAL_PATCH_SIZE)
            && native_extra == expected
            && !Self::has_generic_override(config)
            && config.do_convert_rgb.is_none()
            && config.do_normalize.is_none()
            && config.do_rescale.is_none()
            && config.image_mean.is_none()
            && config.image_std.is_none()
            && config.rescale_factor.is_none()
            && config.size.is_none()
            && config
                .image_processor_type
                .as_deref()
                .is_none_or(|name| name == "Inkling2ImageProcessor")
    }

    fn is_transformers_policy(config: &PreProcessorConfig) -> bool {
        let half = |values: &Option<Vec<f64>>| {
            values
                .as_deref()
                .is_none_or(|values| values == [0.5, 0.5, 0.5])
        };
        let size = HashMap::from([("height".to_string(), 40), ("width".to_string(), 40)]);
        config.image_processor_type.as_deref() == Some("Inkling2ImageProcessor")
            && config.do_convert_rgb.is_none_or(|on| on)
            && config.do_normalize.is_none_or(|on| on)
            && config.do_rescale.is_none_or(|on| on)
            && half(&config.image_mean)
            && half(&config.image_std)
            && config
                .rescale_factor
                .is_none_or(|factor| factor == 1.0 / 255.0)
            && config.size.as_ref().is_none_or(|value| *value == size)
            && config.patch_size.is_none()
            && config.temporal_patch_size.is_none()
            && config.extra.is_empty()
            && !Self::has_generic_override(config)
    }

    fn normalized(raw: f32, channel: usize) -> f32 {
        let scaled = raw / 255.0_f32;
        (scaled - MEAN[channel]) / STD[channel]
    }

    /// The pipeline of both outputs, the float tensor and the wire: `take`
    /// gives the storage of the batch's elements, `level` the elements a raw
    /// value of a channel is written as, `finish` the encoder input.
    fn batch<E: Copy, const N: usize>(
        images: &[&DynamicImage],
        take: impl FnOnce(usize) -> Result<Vec<E>, TransformError>,
        level: impl Fn(f32, usize) -> [E; N],
        finish: impl FnOnce(Vec<E>, usize) -> Result<EncoderInput, TransformError>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let mut counts = Vec::with_capacity(images.len());
        let mut sizes = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len() * 3);
        let mut original = Vec::with_capacity(images.len() * 3);
        let mut total = 0usize;
        for image in images {
            let (width, height) = image.dimensions();
            if width == 0 || height == 0 {
                return Err(TransformError::ShapeError(
                    "Inkling2 image dimensions must be positive".to_string(),
                ));
            }
            let rows = (height as usize).div_ceil(PATCH_SIZE);
            let columns = (width as usize).div_ceil(PATCH_SIZE);
            let count = rows.checked_mul(columns).and_then(|n| total.checked_add(n));
            let Some(next_total) = count.filter(|&n| n <= MAX_PATCHES) else {
                return Err(TransformError::ShapeError(format!(
                    "Inkling2 preprocessing exceeds the {MAX_PATCHES}-patch allocation limit"
                )));
            };
            counts.push(next_total - total);
            sizes.push((width, height));
            grids.extend([1, rows as i64, columns as i64]);
            original.extend([1, i64::from(height), i64::from(width)]);
            total = next_total;
        }

        let mut output = take(N * total * PATCH_VALUES)?;
        let padding = std::array::from_fn::<_, 3, _>(|channel| level(-1.0, channel));
        let table: [[[E; N]; 256]; 3] = std::array::from_fn(|channel| {
            std::array::from_fn(|value| level(value as f32, channel))
        });
        for image in images {
            if let Some(rgba) = image.as_rgba8() {
                // An RGBA image is read where it stands, by a walk of its own:
                // the walk below keeps its shape for an RGB image.
                let rgba = rgba.as_raw().as_chunks::<4>().0;
                let (width, height) = (image.width() as usize, image.height() as usize);
                let mut block = [[[0u8; 4]; PATCH_SIZE]; PATCH_SIZE];
                for patch_y in 0..height.div_ceil(PATCH_SIZE) {
                    for patch_x in 0..width.div_ceil(PATCH_SIZE) {
                        let start = output.len();
                        let columns = patch_x * PATCH_SIZE..((patch_x + 1) * PATCH_SIZE).min(width);
                        let rows = patch_y * PATCH_SIZE..((patch_y + 1) * PATCH_SIZE).min(height);
                        for (row, y) in block.iter_mut().zip(rows) {
                            row[..columns.len()].copy_from_slice(
                                &rgba[y * width + columns.start..y * width + columns.end],
                            );
                        }
                        for y in patch_y * PATCH_SIZE..(patch_y + 1) * PATCH_SIZE {
                            let copied = if y < height { columns.len() } else { 0 };
                            let row = &block[y - patch_y * PATCH_SIZE][..copied];
                            for (x, &[red, green, blue, alpha]) in columns.clone().zip(row) {
                                // An opaque pixel composites to itself.
                                let [red, green, blue] = if alpha == u8::MAX {
                                    [red, green, blue]
                                } else {
                                    let alpha = u32::from(alpha);
                                    let background =
                                        if (x / 8 + y / 8) % 2 == 0 { 204 } else { 153 };
                                    [red, green, blue].map(|value| {
                                        ((u32::from(value) * alpha
                                            + background * (255 - alpha)
                                            + 127)
                                            / 255) as u8
                                    })
                                };
                                output.extend_from_slice(
                                    [
                                        table[0][usize::from(red)],
                                        table[1][usize::from(green)],
                                        table[2][usize::from(blue)],
                                    ]
                                    .as_flattened(),
                                );
                            }
                            for _ in copied..PATCH_SIZE {
                                output.extend_from_slice(padding.as_flattened());
                            }
                        }
                        output.extend_from_within(
                            start..start + N * PATCH_VALUES / TEMPORAL_PATCH_SIZE,
                        );
                    }
                }
                continue;
            }
            // An RGB image is read where it stands: `to_rgba8` would copy it
            // whole at four bytes a pixel.
            let rgb = image.as_rgb8().map(|rgb| rgb.as_raw().as_chunks::<3>().0);
            let rgba = rgb.is_none().then(|| image.to_rgba8());
            let rgba = rgba
                .as_ref()
                .map_or(&[][..], |rgba| rgba.as_raw().as_chunks::<4>().0);
            let (width, height) = (image.width() as usize, image.height() as usize);
            let mut block = [[[0u8; 3]; PATCH_SIZE]; PATCH_SIZE];
            for patch_y in 0..height.div_ceil(PATCH_SIZE) {
                for patch_x in 0..width.div_ceil(PATCH_SIZE) {
                    let start = output.len();
                    // The patch's columns that lie on the image: a run of its
                    // pixels, then padding to the patch's width.
                    let columns = patch_x * PATCH_SIZE..((patch_x + 1) * PATCH_SIZE).min(width);
                    // The patch's pixels are copied out first: a decoder may
                    // leave the frame out of the cache, and these short copies
                    // let its misses overlap.
                    if let Some(rgb) = rgb {
                        let rows = patch_y * PATCH_SIZE..((patch_y + 1) * PATCH_SIZE).min(height);
                        for (row, y) in block.iter_mut().zip(rows) {
                            row[..columns.len()].copy_from_slice(
                                &rgb[y * width + columns.start..y * width + columns.end],
                            );
                        }
                    }
                    for y in patch_y * PATCH_SIZE..(patch_y + 1) * PATCH_SIZE {
                        let on_image = if y < height {
                            y * width + columns.start..y * width + columns.end
                        } else {
                            0..0
                        };
                        let copied = if rgb.is_some() { on_image.len() } else { 0 };
                        for &[red, green, blue] in &block[y - patch_y * PATCH_SIZE][..copied] {
                            output.extend_from_slice(
                                [
                                    table[0][usize::from(red)],
                                    table[1][usize::from(green)],
                                    table[2][usize::from(blue)],
                                ]
                                .as_flattened(),
                            );
                        }
                        let with_alpha = if rgb.is_none() {
                            &rgba[on_image.clone()]
                        } else {
                            &[]
                        };
                        for (x, &[red, green, blue, alpha]) in columns.clone().zip(with_alpha) {
                            // An opaque pixel composites to itself.
                            let [red, green, blue] = if alpha == u8::MAX {
                                [red, green, blue]
                            } else {
                                let alpha = u32::from(alpha);
                                let background = if (x / 8 + y / 8) % 2 == 0 { 204 } else { 153 };
                                [red, green, blue].map(|value| {
                                    ((u32::from(value) * alpha + background * (255 - alpha) + 127)
                                        / 255) as u8
                                })
                            };
                            output.extend_from_slice(
                                [
                                    table[0][usize::from(red)],
                                    table[1][usize::from(green)],
                                    table[2][usize::from(blue)],
                                ]
                                .as_flattened(),
                            );
                        }
                        for _ in on_image.len()..PATCH_SIZE {
                            output.extend_from_slice(padding.as_flattened());
                        }
                    }
                    output
                        .extend_from_within(start..start + N * PATCH_VALUES / TEMPORAL_PATCH_SIZE);
                }
            }
        }
        let tensor = finish(output, total)?;
        let tokens_per_item = counts.iter().map(|&count| count as i64).collect();
        Ok(PreprocessedEncoderInputs::new(tensor, counts, sizes)
            .with_extra(
                "tokens_per_item",
                ModelSpecificValue::int_1d(tokens_per_item),
            )
            .with_extra(
                "vision_grid_thw",
                ModelSpecificValue::int_2d(grids, images.len(), 3),
            )
            .with_extra(
                "vision_original_thw",
                ModelSpecificValue::int_2d(original, images.len(), 3),
            ))
    }
}

/// A refused reserve of the batch's storage, reported alike by both outputs.
fn refused(error: TryReserveError) -> TransformError {
    TransformError::ReserveRefused(format!("Inkling2 patch allocation failed: {error}"))
}

impl VisionPreProcessor for Inkling2ImageProcessor {
    fn default_mean(&self) -> [f64; 3] {
        super::inkling::INKLING_IMAGE_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        super::inkling::INKLING_IMAGE_STD
    }

    /// Transparent pixels are composited over the native checkerboard here,
    /// so the frame must reach the processor with its alpha channel.
    fn keeps_alpha(&self) -> bool {
        true
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Self::batch(
            images,
            |elements| {
                let mut output = Vec::new();
                output.try_reserve_exact(elements).map_err(refused)?;
                Ok(output)
            },
            |raw, channel| [Self::normalized(raw, channel)],
            |output, total| {
                ArrayD::from_shape_vec(IxDyn(&[total, 2, 40, 40, 3]), output)
                    .map(EncoderInput::from)
                    .map_err(|error| TransformError::ShapeError(error.to_string()))
            },
        )
    }

    /// The two narrow float dtypes are written in one pass into a buffer of
    /// the pool: a byte maps to the bytes of its level on the wire, so neither
    /// the float tensor nor the pass that narrows it is made.
    fn preprocess_wire(
        &self,
        images: &[&DynamicImage],
        dtype: EncoderDtype,
    ) -> Option<Result<PreprocessedEncoderInputs, TransformError>> {
        let narrow: fn(f32) -> u16 = match dtype {
            EncoderDtype::BFloat16 => f32_to_bf16_bits,
            EncoderDtype::Float16 => f32_to_f16_bits,
            EncoderDtype::Float32 | EncoderDtype::Uint8 => return None,
        };
        // Mean and std in byte units: a byte's level is this norm's in both
        // narrow dtypes, and not in float32.
        let norm = PixelNorm {
            scale: 1.0,
            offset: MEAN.map(|mean| mean * 255.0),
            divisor: STD.map(|std| std * 255.0),
            channel_block: 1,
        };
        Some(Self::batch(
            images,
            |elements| buffer_pool::take_empty(elements).map_err(refused),
            |raw, channel| narrow(Self::normalized(raw, channel)).to_le_bytes(),
            |wire, total| {
                EncoderInput::wire(wire, dtype, vec![total, 2, 40, 40, 3], norm)
                    .map_err(TransformError::ShapeError)
            },
        ))
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        if width == 0 || height == 0 {
            return Err(TransformError::ShapeError(
                "Inkling2 image dimensions must be positive".to_string(),
            ));
        }
        Ok((height as usize).div_ceil(PATCH_SIZE) * (width as usize).div_ceil(PATCH_SIZE))
    }

    fn model_name(&self) -> &'static str {
        "inkling2"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage, Rgba, RgbaImage};
    use serde_json::json;

    use super::*;
    use crate::vision::processor::VideoClipOptions;

    fn config() -> PreProcessorConfig {
        PreProcessorConfig::from_value(json!({
            "patch_size": 40, "temporal_patch_size": 2,
            "include_extra_column": false,
            "rescale_image_frac": null, "rescale_image_max_upscaled_long_edge": null,
            "stretch_to_padded_grid": false,
            "overlay_rgba_images_on_checkerboard": true, "requires_image_grid_thw": true
        }))
        .expect("native configuration")
    }

    fn pixels(width: u32, height: u32, alpha: bool) -> DynamicImage {
        let value = |x: u32, y: u32, channel: u32| ((x * 17 + y * 31 + channel * 67) % 256) as u8;
        if alpha {
            DynamicImage::ImageRgba8(RgbaImage::from_fn(width, height, |x, y| {
                Rgba(std::array::from_fn(|c| value(x, y, c as u32)))
            }))
        } else {
            DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
                Rgb(std::array::from_fn(|c| value(x, y, c as u32)))
            }))
        }
    }

    fn bf16_hash(values: &[f32]) -> String {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for value in values {
            let bits = value.to_bits();
            let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
            for byte in rounded.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        format!("{hash:016x}")
    }

    fn integers(output: &PreprocessedEncoderInputs, name: &str) -> Vec<i64> {
        match &output.model_specific[name] {
            ModelSpecificValue::IntTensor { data, shape } => {
                assert_eq!(shape, &[output.batch_size(), 3]);
                data.clone()
            }
            other => panic!("expected integer grid, got {other:?}"),
        }
    }

    #[test]
    fn inkling2_native_bf16_matches_pinned_python_oracle() {
        // Pinned image_processing.py SHA256 c7698d4942314a00afb82de553339454a
        // aa9afa67df40cc891d4e47b619452ac; generated with its actual preprocess().
        let cases = [
            (1, 1, false, "7c93dbb9eb757061"),
            (39, 39, false, "e825d7b5284b407d"),
            (40, 40, false, "3060ce5f78fa16ed"),
            (41, 41, false, "6f7dcc8f1ecb8249"),
            (80, 80, false, "75561f363c664ee1"),
            (41, 39, false, "88f3cd8c9010817d"),
            (39, 41, false, "1c45f4a48eaae715"),
            (80, 40, false, "b82b747d5f838d45"),
            (17, 9, true, "740a53642f7bee95"),
            (41, 39, true, "571dd96bce3882c1"),
        ];
        let processor = Inkling2ImageProcessor::new();
        for (width, height, alpha, expected) in cases {
            let output = processor
                .preprocess(&[&pixels(width, height, alpha)])
                .expect("native image");
            let rows = height.div_ceil(40);
            let columns = width.div_ceil(40);
            let count = (rows * columns) as usize;
            assert_eq!(output.encoder_input.shape(), &[count, 2, 40, 40, 3]);
            assert_eq!(output.feature_token_counts, [count]);
            assert_eq!(output.item_sizes, [(width, height)]);
            assert_eq!(
                integers(&output, "vision_grid_thw"),
                [1, i64::from(rows), i64::from(columns)]
            );
            assert_eq!(
                integers(&output, "vision_original_thw"),
                [1, i64::from(height), i64::from(width)]
            );
            assert_eq!(
                processor.calculate_num_tokens(width, height).unwrap(),
                count
            );
            let data = output.encoder_input.flat_f32();
            assert_eq!(
                bf16_hash(&data),
                expected,
                "{width}x{height}, alpha={alpha}"
            );
            for patch in data.as_chunks::<PATCH_VALUES>().0 {
                assert_eq!(&patch[..PATCH_VALUES / 2], &patch[PATCH_VALUES / 2..]);
            }
        }
    }

    #[test]
    fn inkling2_multiple_images_keep_grid_and_feature_order() {
        let output = Inkling2ImageProcessor::new()
            .preprocess(&[&pixels(41, 39, false), &pixels(17, 9, true)])
            .expect("multiple native images");
        assert_eq!(output.feature_token_counts, [2, 1]);
        match &output.model_specific["tokens_per_item"] {
            ModelSpecificValue::IntTensor { data, shape } => {
                assert_eq!(data, &[2, 1]);
                assert_eq!(shape, &[2]);
            }
            other => panic!("expected per-image counts, got {other:?}"),
        }
        assert_eq!(integers(&output, "vision_grid_thw"), [1, 1, 2, 1, 1, 1]);
        assert_eq!(
            integers(&output, "vision_original_thw"),
            [1, 39, 41, 1, 9, 17]
        );
        let data = output.encoder_input.flat_f32();
        assert_eq!(bf16_hash(&data[..2 * PATCH_VALUES]), "88f3cd8c9010817d");
        assert_eq!(bf16_hash(&data[2 * PATCH_VALUES..]), "740a53642f7bee95");
    }

    #[test]
    fn inkling2_checkerboard_uses_integer_rounding_and_negative_raw_padding() {
        let mut image = RgbaImage::new(9, 9);
        image.put_pixel(1, 0, Rgba([10, 20, 30, 255]));
        image.put_pixel(8, 0, Rgba([255, 0, 51, 128]));
        let output = Inkling2ImageProcessor::new()
            .preprocess(&[&DynamicImage::ImageRgba8(image)])
            .expect("transparent image");
        let data = output.encoder_input.flat_f32();
        for (x, y, rgb) in [
            (0, 0, [204, 204, 204]),
            (1, 0, [10, 20, 30]),
            (8, 0, [204, 76, 102]),
            (0, 8, [153, 153, 153]),
            (8, 8, [204, 204, 204]),
            (9, 0, [-1, -1, -1]),
        ] {
            for channel in 0..3 {
                let scaled = rgb[channel] as f32 / 255.0_f32;
                let expected = (scaled - MEAN[channel]) / STD[channel];
                assert_eq!(
                    data[(y * 40 + x) * 3 + channel].to_bits(),
                    expected.to_bits()
                );
            }
        }
    }

    #[test]
    fn inkling2_rgba_walk_agrees_with_the_rgb_walk_and_the_converted_copy() {
        // Three patch rows with ragged edges. One pixel past the first patch
        // row is partly transparent, where the checkerboard phase by the
        // image's row and by the row inside the patch differ.
        let rgb = pixels(81, 81, false);
        let mut rgba = rgb.to_rgba8();
        rgba.get_pixel_mut(9, 48).0[3] = 128;
        let deep = image::ImageBuffer::from_fn(81, 81, |x, y| {
            Rgba(rgba.get_pixel(x, y).0.map(|value| u16::from(value) * 257))
        });
        let bits = |image: DynamicImage| -> Vec<u32> {
            let output = Inkling2ImageProcessor::new()
                .preprocess(&[&image])
                .expect("native image");
            let data = output.encoder_input.flat_f32();
            data.iter().map(|value| value.to_bits()).collect()
        };
        let mut expected = bits(rgb);
        // (105, 172, 239) at alpha 128 over the background 153.
        for (channel, value) in [129_u8, 163, 196].into_iter().enumerate() {
            let level = ((f32::from(value) / 255.0_f32 - MEAN[channel]) / STD[channel]).to_bits();
            let at = 3 * PATCH_VALUES + (8 * PATCH_SIZE + 9) * 3 + channel;
            expected[at] = level;
            expected[at + PATCH_VALUES / 2] = level;
        }
        let in_place = bits(DynamicImage::ImageRgba8(rgba));
        assert_eq!(in_place, expected);
        assert_eq!(bits(DynamicImage::ImageRgba16(deep)), in_place);
    }

    #[test]
    fn inkling2_rejects_missing_native_policy_and_unsupported_overrides() {
        assert!(Inkling2ImageProcessor::validate_config(&PreProcessorConfig::default()).is_err());
        assert!(
            Inkling2ImageProcessor::from_preprocessor_config(&PreProcessorConfig::default())
                .is_err()
        );
        for (key, value) in [
            ("include_extra_column", json!(true)),
            ("rescale_image_frac", json!(1.0)),
            ("rescale_image_max_upscaled_long_edge", json!(2048)),
            ("stretch_to_padded_grid", json!(true)),
            ("overlay_rgba_images_on_checkerboard", json!(false)),
            ("requires_image_grid_thw", json!(false)),
            ("virtual_width", json!(80)),
        ] {
            let mut config = config();
            config.extra.insert(key.to_string(), value);
            assert!(
                Inkling2ImageProcessor::validate_config(&config).is_err(),
                "{key}"
            );
        }
        let mut wrong = config();
        wrong.do_resize = Some(true);
        assert!(Inkling2ImageProcessor::validate_config(&wrong).is_err());
        wrong = config();
        wrong.patch_size.as_mut().expect("patch size").width = Some(20);
        assert!(Inkling2ImageProcessor::validate_config(&wrong).is_err());
    }

    /// The `image_processor` section of the released `processor_config.json`.
    fn released_section() -> Value {
        json!({
            "do_convert_rgb": true,
            "do_normalize": true,
            "do_rescale": true,
            "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "Inkling2ImageProcessor",
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 0.00392156862745098,
            "size": {"height": 40, "width": 40}
        })
    }

    #[test]
    fn inkling2_accepts_the_released_transformers_section() {
        let section = PreProcessorConfig::from_value(released_section()).expect("section");
        Inkling2ImageProcessor::from_preprocessor_config(&section).unwrap();
        // Transformers fills a field the section leaves out from the class
        // defaults, which are the same values.
        let minimal = PreProcessorConfig::from_value(
            json!({"image_processor_type": "Inkling2ImageProcessor"}),
        )
        .expect("minimal section");
        Inkling2ImageProcessor::validate_config(&minimal).unwrap();
    }

    #[test]
    fn inkling2_rejects_changed_transformers_sections() {
        for (key, value) in [
            ("image_processor_type", json!("CLIPImageProcessor")),
            ("do_convert_rgb", json!(false)),
            ("do_normalize", json!(false)),
            ("do_rescale", json!(false)),
            ("image_mean", json!([0.48145466, 0.4578275, 0.40821073])),
            ("image_std", json!([0.26862954, 0.26130258, 0.27577711])),
            ("rescale_factor", json!(1.0)),
            ("size", json!({"height": 20, "width": 20})),
            ("do_resize", json!(true)),
            ("patch_size", json!(40)),
            ("temporal_patch_size", json!(2)),
            ("overlay_rgba_images_on_checkerboard", json!(true)),
        ] {
            let mut section = released_section();
            section[key] = value;
            let config = PreProcessorConfig::from_value(section).expect("section");
            assert!(
                Inkling2ImageProcessor::validate_config(&config).is_err(),
                "{key}"
            );
        }
        let mut untyped = released_section();
        untyped
            .as_object_mut()
            .expect("object")
            .remove("image_processor_type");
        let config = PreProcessorConfig::from_value(untyped).expect("section");
        assert!(Inkling2ImageProcessor::validate_config(&config).is_err());
    }

    #[test]
    fn inkling2_native_policy_rejects_transformers_fields() {
        let mut mixed = config();
        mixed.image_mean = Some(vec![0.5, 0.5, 0.5]);
        assert!(Inkling2ImageProcessor::validate_config(&mixed).is_err());
    }

    #[test]
    fn inkling2_empty_batch_and_invalid_dimensions_are_explicit() {
        let processor = Inkling2ImageProcessor::from_preprocessor_config(&config()).unwrap();
        let empty = processor.preprocess(&[]).expect("native empty batch");
        assert_eq!(empty.encoder_input.shape(), &[0, 2, 40, 40, 3]);
        assert!(integers(&empty, "vision_grid_thw").is_empty());
        assert!(processor.calculate_num_tokens(0, 1).is_err());
        assert!(processor.preprocess(&[&pixels(0, 1, false)]).is_err());
        assert!(processor
            .preprocess_video(&[pixels(1, 1, false)], &VideoClipOptions::sampled_at(1.0))
            .is_err());
        assert!(processor.keeps_alpha());
    }

    #[test]
    fn inkling2_wire_is_the_float_tensor_written_in_both_narrow_dtypes() {
        let processor = Inkling2ImageProcessor::from_preprocessor_config(&config()).unwrap();
        let images = [&pixels(83, 121, false), &pixels(57, 80, true)];
        let two_pass = processor.preprocess(&images).unwrap();
        for dtype in [EncoderDtype::BFloat16, EncoderDtype::Float16] {
            let narrow = |level: f32| match dtype {
                EncoderDtype::BFloat16 => f32_to_bf16_bits(level),
                _ => f32_to_f16_bits(level),
            };
            let wire = processor.preprocess_wire(&images, dtype).unwrap().unwrap();
            assert_eq!(wire.feature_token_counts, two_pass.feature_token_counts);
            assert_eq!(
                integers(&wire, "vision_grid_thw"),
                integers(&two_pass, "vision_grid_thw")
            );
            assert_eq!(wire.encoder_input.shape(), two_pass.encoder_input.shape());
            // The norm the wire names gives every byte its level.
            let levels = wire.encoder_input.pixel_norm().unwrap().table_f32();
            for (channel, levels) in levels.iter().enumerate() {
                for (value, &level) in levels.iter().enumerate() {
                    assert_eq!(
                        narrow(level),
                        narrow(Inkling2ImageProcessor::normalized(value as f32, channel)),
                        "{dtype} channel {channel} byte {value}"
                    );
                }
            }
            assert!(
                wire.encoder_input.into_bytes_as(dtype).unwrap()
                    == two_pass.encoder_input.write_as(dtype),
                "{dtype}: the wire differs from the float tensor's bytes"
            );
        }
        assert!(processor
            .preprocess_wire(&images, EncoderDtype::Float32)
            .is_none());
    }
}
