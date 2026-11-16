//! Inkling2 native still-image preprocessing, without training-only virtual resizing.
//!
//! Features retain separate F32 divide/subtract/divide normalization until the
//! common BF16 wire conversion. Grid cells are pixel-major RGB patches with two
//! identical temporal slices; every cell produces one language-model token.

use std::collections::HashMap;

use image::{DynamicImage, GenericImageView};
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Value};

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::TransformError,
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
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Result<Self, TransformError> {
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

        let mut output = Vec::new();
        output
            .try_reserve_exact(total * PATCH_VALUES)
            .map_err(|error| {
                TransformError::ShapeError(format!("Inkling2 patch allocation failed: {error}"))
            })?;
        let padding = std::array::from_fn::<_, 3, _>(|channel| Self::normalized(-1.0, channel));
        for image in images {
            let rgba = image.to_rgba8();
            let (width, height) = (rgba.width() as usize, rgba.height() as usize);
            let raw = rgba.as_raw();
            for patch_y in 0..height.div_ceil(PATCH_SIZE) {
                for patch_x in 0..width.div_ceil(PATCH_SIZE) {
                    let start = output.len();
                    for y in patch_y * PATCH_SIZE..(patch_y + 1) * PATCH_SIZE {
                        for x in patch_x * PATCH_SIZE..(patch_x + 1) * PATCH_SIZE {
                            if y >= height || x >= width {
                                output.extend_from_slice(&padding);
                                continue;
                            }
                            let pixel = &raw[(y * width + x) * 4..][..4];
                            let alpha = u32::from(pixel[3]);
                            let background = if (x / 8 + y / 8) % 2 == 0 { 204 } else { 153 };
                            for (channel, value) in pixel[..3].iter().enumerate() {
                                let value =
                                    (u32::from(*value) * alpha + background * (255 - alpha) + 127)
                                        / 255;
                                output.push(Self::normalized(value as f32, channel));
                            }
                        }
                    }
                    output.extend_from_within(start..start + PATCH_VALUES / TEMPORAL_PATCH_SIZE);
                }
            }
        }
        let tensor = ArrayD::from_shape_vec(IxDyn(&[total, 2, 40, 40, 3]), output)
            .map_err(|error| TransformError::ShapeError(error.to_string()))?;
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
}
