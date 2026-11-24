//! Inkling image processor.
//!
//! Implements `InklingImageProcessor`: optional aspect-preserving
//! resize, RGB HWC input, CLIP normalization, and one extra patch column on
//! every row. The output tensor is
//! `[sum_patches, temporal_patch_size, patch_size, patch_size, 3]`.

use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{ArrayD, IxDyn};

use crate::{
    buffer_pool,
    encoder_inputs::{f32_to_bf16_bits, EncoderDtype, EncoderInput, PixelNorm},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::{self, TransformError},
    },
};

pub(crate) const INKLING_IMAGE_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
pub(crate) const INKLING_IMAGE_STD: [f64; 3] = [0.26862954, 0.2613026, 0.2757771];
pub(crate) const DEFAULT_PATCH_SIZE: usize = 40;
pub(crate) const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;
pub(crate) const DEFAULT_RESCALE_IMAGE_MAX_UPSCALED_LONG_EDGE: u32 = 2048;
const PAD_RAW_VALUE: f32 = -1.0 / 255.0;

/// Default cap on the total number of patches materialized for one preprocess
/// call. Each patch is `temporal(2)*patch*patch*3` f32 (≈38 KiB at the default
/// patch size), so this bounds the up-front tensor reservation. Without a cap, a
/// small crafted image that decodes to huge dimensions drives a multi-GB
/// allocation that aborts the process. Override via the `in_patch_limit`
/// preprocessor-config key.
pub(crate) const DEFAULT_INKLING_MAX_PATCHES: usize = 32_768;

#[derive(Debug, Clone)]
pub struct InklingImageProcessor {
    patch_size: usize,
    temporal_patch_size: usize,
    rescale_image_frac: Option<f64>,
    rescale_image_max_upscaled_long_edge: Option<u32>,
    /// Upper bound on total patches per preprocess call; `None` disables it.
    max_patches: Option<usize>,
    mean: [f64; 3],
    std: [f64; 3],
}

impl Default for InklingImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl InklingImageProcessor {
    pub fn new() -> Self {
        Self {
            patch_size: DEFAULT_PATCH_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            // Count patches at the authored image dimensions. Resizing remains
            // an explicit preprocessor override.
            rescale_image_frac: None,
            rescale_image_max_upscaled_long_edge: Some(
                DEFAULT_RESCALE_IMAGE_MAX_UPSCALED_LONG_EDGE,
            ),
            max_patches: Some(DEFAULT_INKLING_MAX_PATCHES),
            mean: INKLING_IMAGE_MEAN,
            std: INKLING_IMAGE_STD,
        }
    }

    /// Resolve `config` over the defaults once; a config whose geometry or
    /// rescale the processor cannot serve is refused here.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Result<Self, TransformError> {
        let processor = Self::new().with_preprocessor_config(config);
        processor.validate()?;
        Ok(processor)
    }

    fn with_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        let mut processor = self.clone();
        if config.patch_size.is_some() {
            processor.patch_size = config.get_patch_size(self.patch_size);
        }
        if let Some(temporal_patch_size) = config.temporal_patch_size {
            processor.temporal_patch_size = temporal_patch_size;
        }
        if let Some(value) = config.extra.get("rescale_image_frac") {
            processor.rescale_image_frac = value.as_f64();
        }
        if let Some(value) = config.extra.get("rescale_image_max_upscaled_long_edge") {
            processor.rescale_image_max_upscaled_long_edge =
                value.as_u64().and_then(|v| u32::try_from(v).ok());
        }
        // Only override on a well-formed value; a missing/malformed key keeps the
        // default cap rather than silently disabling the budget.
        if let Some(limit) = config
            .extra
            .get("in_patch_limit")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
        {
            processor.max_patches = Some(limit);
        }
        processor.mean = config
            .image_mean
            .as_ref()
            .map_or(INKLING_IMAGE_MEAN, |values| {
                if values.len() >= 3 {
                    [values[0], values[1], values[2]]
                } else {
                    INKLING_IMAGE_MEAN
                }
            });
        processor.std = config
            .image_std
            .as_ref()
            .map_or(INKLING_IMAGE_STD, |values| {
                if values.len() >= 3 {
                    [values[0], values[1], values[2]]
                } else {
                    INKLING_IMAGE_STD
                }
            });
        processor
    }

    fn patch_grid(&self, width: usize, height: usize) -> (usize, usize, usize) {
        let nph = height.div_ceil(self.patch_size);
        let npw = width / self.patch_size + 1;
        let num_patches = nph * npw;
        (nph, npw, num_patches)
    }

    fn num_patch_features(&self) -> usize {
        self.temporal_patch_size * self.patch_size * self.patch_size * 3
    }

    /// Reject a preprocess whose total patch count would drive an oversized
    /// tensor allocation, *before* any memory is reserved.
    fn ensure_patch_budget(&self, total_patches: usize) -> Result<(), TransformError> {
        if let Some(limit) = self.max_patches {
            if total_patches > limit {
                return Err(TransformError::ShapeError(format!(
                    "Inkling image is too large: {total_patches} patches exceed the limit of \
                     {limit} (raise the `in_patch_limit` preprocessor setting to allow it)"
                )));
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), TransformError> {
        if self.patch_size == 0 || self.temporal_patch_size == 0 {
            return Err(TransformError::ShapeError(
                "Inkling patch_size and temporal_patch_size must be positive".to_string(),
            ));
        }
        if let Some(frac) = self.rescale_image_frac {
            if !frac.is_finite() || frac <= 0.0 {
                return Err(TransformError::ShapeError(format!(
                    "Inkling rescale_image_frac must be positive and finite, got {frac}"
                )));
            }
        }
        if matches!(self.rescale_image_max_upscaled_long_edge, Some(0)) {
            return Err(TransformError::ShapeError(
                "Inkling rescale_image_max_upscaled_long_edge must be positive".to_string(),
            ));
        }
        if matches!(self.max_patches, Some(0)) {
            return Err(TransformError::ShapeError(
                "Inkling max_patches must be positive".to_string(),
            ));
        }
        Ok(())
    }

    fn scaled_image_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        let Some(frac) = self.rescale_image_frac else {
            return (width, height);
        };
        let long_edge = width.max(height);
        if long_edge == 0 {
            return (width, height);
        }

        let mut target_long_edge = long_edge as f64 * frac;
        if let Some(max_upscaled_long_edge) = self.rescale_image_max_upscaled_long_edge {
            let effective_cap = max_upscaled_long_edge.max(long_edge);
            target_long_edge = target_long_edge.min(effective_cap as f64);
        }
        let ratio = target_long_edge / long_edge as f64;
        if (ratio - 1.0).abs() < f64::EPSILON {
            return (width, height);
        }

        let scale_dim = |dim: u32| -> u32 { ((dim as f64 * ratio + 0.5).floor() as u32).max(1) };
        (scale_dim(width), scale_dim(height))
    }

    fn prepare_rgb_image(&self, image: &DynamicImage) -> RgbImage {
        let (width, height) = image.dimensions();
        let (scaled_width, scaled_height) = self.scaled_image_dimensions(width, height);
        if (scaled_width, scaled_height) == (width, height) {
            image.to_rgb8()
        } else {
            transforms::resize_lanczos_pil(image, scaled_width, scaled_height).into_rgb8()
        }
    }

    /// The float a raw value of a channel is normalized to.
    fn normalized(&self) -> impl Fn(f32, usize) -> f32 {
        // Fused: (raw/255 - mean) / std = raw * (1/(255*std)) - mean/std.
        let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * self.std[c] as f32));
        let bias: [f32; 3] = std::array::from_fn(|c| -(self.mean[c] as f32) / (self.std[c] as f32));
        move |raw, channel| raw * scale[channel] + bias[channel]
    }

    fn append_image_patches<E: Copy, const N: usize>(
        &self,
        image: &DynamicImage,
        level: &impl Fn(f32, usize) -> [E; N],
        output: &mut Vec<E>,
    ) {
        let rgb = self.prepare_rgb_image(image);
        let width = rgb.width() as usize;
        let height = rgb.height() as usize;
        let raw = rgb.as_raw();
        let (_, npw, _) = self.patch_grid(width, height);
        // PAD_RAW_VALUE is in the [0,1] domain; recover the raw byte for scale/bias.
        let pad_byte = PAD_RAW_VALUE * 255.0;
        let pad_norm: [[E; N]; 3] = std::array::from_fn(|c| level(pad_byte, c));
        let table: [[[E; N]; 256]; 3] =
            std::array::from_fn(|c| std::array::from_fn(|byte| level(byte as f32, c)));
        let (pixels, _) = raw.as_chunks::<3>();

        for patch_idx in 0..height.div_ceil(self.patch_size) * npw {
            let patch_y = patch_idx / npw;
            let patch_x = patch_idx - patch_y * npw;
            let y_base = patch_y * self.patch_size;
            let x_base = patch_x * self.patch_size;
            let patch_start = output.len();

            // The patch's columns that lie on the image: a run of its pixels,
            // then padding to the patch's width.
            let columns = x_base.min(width)..(x_base + self.patch_size).min(width);
            for iy in y_base..y_base + self.patch_size {
                let on_image = if iy < height {
                    &pixels[iy * width..][columns.clone()]
                } else {
                    &[]
                };
                for pixel in on_image {
                    output.extend_from_slice(
                        [
                            table[0][usize::from(pixel[0])],
                            table[1][usize::from(pixel[1])],
                            table[2][usize::from(pixel[2])],
                        ]
                        .as_flattened(),
                    );
                }
                for _ in on_image.len()..self.patch_size {
                    output.extend_from_slice(pad_norm.as_flattened());
                }
            }

            let single_temporal_patch_len = N * self.patch_size * self.patch_size * 3;
            for _ in 1..self.temporal_patch_size {
                output.extend_from_within(patch_start..patch_start + single_temporal_patch_len);
            }
        }
    }

    /// The pipeline of both outputs, the float tensor and the wire: `take`
    /// gives the storage of the batch's elements, `level` the elements a raw
    /// value of a channel is written as, `finish` the encoder input.
    fn batch<E: Copy, const N: usize>(
        &self,
        images: &[&DynamicImage],
        take: impl FnOnce(usize) -> Result<Vec<E>, TransformError>,
        level: impl Fn(f32, usize) -> [E; N],
        finish: impl FnOnce(Vec<E>, usize) -> Result<EncoderInput, TransformError>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        TransformError::if_empty(images)?;

        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut tokens_per_item = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut total_patches = 0usize;
        for image in images {
            let (width, height) = image.dimensions();
            let (scaled_width, scaled_height) = self.scaled_image_dimensions(width, height);
            let (_, _, num_patches) =
                self.patch_grid(scaled_width as usize, scaled_height as usize);
            feature_token_counts.push(num_patches);
            tokens_per_item.push(num_patches as i64);
            item_sizes.push((scaled_width, scaled_height));
            total_patches += num_patches;
        }

        // Enforce the patch budget BEFORE reserving. A small crafted image can
        // decode to huge dimensions; the reservation below would otherwise abort
        // the whole process (with_capacity/handle_alloc_error is uncatchable).
        self.ensure_patch_budget(total_patches)?;
        let mut patches = take(N * total_patches * self.num_patch_features())?;
        for image in images {
            self.append_image_patches(image, &level, &mut patches);
        }
        let encoder_input = finish(patches, total_patches)?;

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra(
                    "tokens_per_item",
                    ModelSpecificValue::int_1d(tokens_per_item),
                ),
        )
    }

    /// The encoder input's shape for `total_patches` patches.
    fn shape(&self, total_patches: usize) -> Vec<usize> {
        vec![
            total_patches,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
            3,
        ]
    }
}

impl VisionPreProcessor for InklingImageProcessor {
    fn default_mean(&self) -> [f64; 3] {
        INKLING_IMAGE_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        INKLING_IMAGE_STD
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let normalized = self.normalized();
        self.batch(
            images,
            |capacity| {
                let mut patches: Vec<f32> = Vec::new();
                patches.try_reserve_exact(capacity).map_err(|e| {
                    TransformError::ReserveRefused(format!(
                        "Inkling: cannot reserve {capacity} f32 for image patches: {e}"
                    ))
                })?;
                Ok(patches)
            },
            |raw, channel| [normalized(raw, channel)],
            |patches, total_patches| {
                ArrayD::from_shape_vec(IxDyn(&self.shape(total_patches)), patches)
                    .map(EncoderInput::from)
                    .map_err(|e| {
                        TransformError::ShapeError(format!(
                            "failed to create Inkling image encoder input {:?}: {e}",
                            self.shape(total_patches)
                        ))
                    })
            },
        )
    }

    /// A bfloat16 wire is written in one pass into a buffer of the pool: a
    /// byte maps to the two bytes of its level on the wire, so neither the
    /// float tensor nor the pass that narrows it is made.
    fn preprocess_wire(
        &self,
        images: &[&DynamicImage],
        dtype: EncoderDtype,
    ) -> Option<Result<PreprocessedEncoderInputs, TransformError>> {
        // float16 keeps the two passes: under the default constants one of
        // its levels is not the one the norm below gives.
        if dtype != EncoderDtype::BFloat16 {
            return None;
        }
        let normalized = self.normalized();
        // Mean and std in byte units, the norm the wire names. The fused
        // levels round apart from it near zero and the config sets mean and
        // std, so the one pass is taken only where every byte narrows to the
        // norm's own level.
        let norm = PixelNorm {
            scale: 1.0,
            offset: self.mean.map(|mean| mean as f32 * 255.0),
            divisor: self.std.map(|std| std as f32 * 255.0),
            channel_block: 1,
        };
        let named = norm.table_f32();
        let agree = (0..3).all(|channel| {
            (0..256).all(|value| {
                f32_to_bf16_bits(named[channel][value])
                    == f32_to_bf16_bits(normalized(value as f32, channel))
            })
        });
        if !agree {
            return None;
        }
        Some(self.batch(
            images,
            |elements| {
                // Empty: the patches are appended, so no byte of a previous
                // request can be part of the wire.
                buffer_pool::take_empty(elements).map_err(|e| {
                    TransformError::ReserveRefused(format!(
                        "Inkling: cannot reserve {elements} bytes for image patches: {e}"
                    ))
                })
            },
            |raw, channel| f32_to_bf16_bits(normalized(raw, channel)).to_le_bytes(),
            |wire, total_patches| {
                EncoderInput::wire(wire, dtype, self.shape(total_patches), norm)
                    .map_err(TransformError::ShapeError)
            },
        ))
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let (scaled_width, scaled_height) = self.scaled_image_dimensions(width, height);
        Ok(self
            .patch_grid(scaled_width as usize, scaled_height as usize)
            .2)
    }

    fn model_name(&self) -> &'static str {
        "inkling"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::{preprocessor_config::PatchSize, processor::ModelSpecificValue};

    fn test_image() -> DynamicImage {
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([0, 127, 255]));
        img.put_pixel(1, 0, Rgb([255, 127, 0]));
        img.put_pixel(0, 1, Rgb([10, 20, 30]));
        img.put_pixel(1, 1, Rgb([40, 50, 60]));
        DynamicImage::ImageRgb8(img)
    }

    fn processor_without_rescale() -> InklingImageProcessor {
        InklingImageProcessor {
            patch_size: DEFAULT_PATCH_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            rescale_image_frac: None,
            rescale_image_max_upscaled_long_edge: None,
            max_patches: None,
            mean: INKLING_IMAGE_MEAN,
            std: INKLING_IMAGE_STD,
        }
    }

    #[test]
    fn ensure_patch_budget_enforces_limit_and_none_disables() {
        let mut processor = InklingImageProcessor::new();
        processor.max_patches = Some(10);
        assert!(processor.ensure_patch_budget(10).is_ok());
        assert!(processor.ensure_patch_budget(11).is_err());
        processor.max_patches = None;
        assert!(processor.ensure_patch_budget(usize::MAX).is_ok());
    }

    #[test]
    fn preprocess_rejects_image_exceeding_in_patch_limit() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(2),
                width: Some(2),
            }),
            temporal_patch_size: Some(2),
            // test_image() is 2x2 → 2 patches at patch_size 2, over the limit of 1.
            extra: std::collections::HashMap::from([(
                "in_patch_limit".to_string(),
                serde_json::json!(1),
            )]),
            ..PreProcessorConfig::default()
        };
        let processor = InklingImageProcessor::from_preprocessor_config(&config).unwrap();
        let err = processor
            .preprocess(&[&test_image()])
            .expect_err("must reject an image exceeding in_patch_limit");
        assert!(
            matches!(err, TransformError::ShapeError(_)),
            "expected ShapeError, got {err:?}"
        );
    }

    #[test]
    fn preprocess_succeeds_within_in_patch_limit() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(2),
                width: Some(2),
            }),
            temporal_patch_size: Some(2),
            extra: std::collections::HashMap::from([(
                "in_patch_limit".to_string(),
                serde_json::json!(8),
            )]),
            ..PreProcessorConfig::default()
        };
        let processor = InklingImageProcessor::from_preprocessor_config(&config).unwrap();
        assert!(processor.preprocess(&[&test_image()]).is_ok());
    }

    fn norm(raw: f32, channel: usize) -> f32 {
        ((raw / 255.0) as f64 - INKLING_IMAGE_MEAN[channel]) as f32
            / INKLING_IMAGE_STD[channel] as f32
    }

    fn pad_norm(channel: usize) -> f32 {
        (PAD_RAW_VALUE as f64 - INKLING_IMAGE_MEAN[channel]) as f32
            / INKLING_IMAGE_STD[channel] as f32
    }

    #[test]
    fn inkling_image_shape_includes_extra_patch_column_and_temporal_duplication() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(2),
                width: Some(2),
            }),
            temporal_patch_size: Some(2),
            ..PreProcessorConfig::default()
        };
        let processor = processor_without_rescale().with_preprocessor_config(&config);
        let result = processor.preprocess(&[&test_image()]).unwrap();
        assert_eq!(result.encoder_input.shape(), &[2, 2, 2, 2, 3]);
        assert_eq!(result.feature_token_counts, vec![2]);

        let flat = result.encoder_input.flat_f32().into_owned();
        assert!((flat[0] - norm(0.0, 0)).abs() < 1e-6);
        assert!((flat[1] - norm(127.0, 1)).abs() < 1e-6);
        assert!((flat[2] - norm(255.0, 2)).abs() < 1e-6);

        let single_temporal_len = 2 * 2 * 3;
        assert_eq!(
            &flat[0..single_temporal_len],
            &flat[single_temporal_len..2 * single_temporal_len]
        );

        let second_patch_start = 2 * single_temporal_len;
        assert!((flat[second_patch_start] - pad_norm(0)).abs() < 1e-6);
        assert!((flat[second_patch_start + 1] - pad_norm(1)).abs() < 1e-6);
        assert!((flat[second_patch_start + 2] - pad_norm(2)).abs() < 1e-6);
    }

    #[test]
    fn calculate_num_tokens_matches_expected_grid_rule() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(40),
                width: Some(40),
            }),
            ..PreProcessorConfig::default()
        };
        let processor = processor_without_rescale().with_preprocessor_config(&config);
        assert_eq!(processor.calculate_num_tokens(80, 40).unwrap(), 3);
        assert_eq!(processor.calculate_num_tokens(81, 41).unwrap(), 6);
    }

    #[test]
    fn calculate_num_tokens_uses_authored_dimensions_by_default() {
        let processor = InklingImageProcessor::new();
        assert_eq!(processor.calculate_num_tokens(640, 480).unwrap(), 204);
        assert_eq!(processor.calculate_num_tokens(1200, 600).unwrap(), 465);
        assert_eq!(processor.calculate_num_tokens(3000, 2000).unwrap(), 3800);
    }

    #[test]
    fn config_can_explicitly_enable_image_rescale() {
        let config = PreProcessorConfig::from_json(
            r#"{"patch_size":40,"rescale_image_frac":2.0,"rescale_image_max_upscaled_long_edge":2048}"#,
        )
        .unwrap();
        let processor = InklingImageProcessor::from_preprocessor_config(&config).unwrap();

        assert_eq!(processor.calculate_num_tokens(640, 480).unwrap(), 792);
    }

    #[test]
    fn preprocess_preserves_authored_size_by_default() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(2),
                width: Some(2),
            }),
            temporal_patch_size: Some(2),
            ..PreProcessorConfig::default()
        };
        let processor = InklingImageProcessor::from_preprocessor_config(&config).unwrap();

        let result = processor.preprocess(&[&test_image()]).unwrap();

        assert_eq!(result.encoder_input.shape(), &[2, 2, 2, 2, 3]);
        assert_eq!(result.feature_token_counts, vec![2]);
        assert_eq!(result.item_sizes, vec![(2, 2)]);
        assert!(matches!(
            result.model_specific.get("tokens_per_item"),
            Some(ModelSpecificValue::IntTensor { data, shape })
                if data.as_slice() == [2] && shape.as_slice() == [1]
        ));
    }

    #[test]
    fn preprocess_applies_explicit_resize_before_patchifying() {
        let config = PreProcessorConfig::from_json(
            r#"{"patch_size":2,"temporal_patch_size":2,"rescale_image_frac":2.0}"#,
        )
        .unwrap();
        let processor = InklingImageProcessor::from_preprocessor_config(&config).unwrap();

        let result = processor.preprocess(&[&test_image()]).unwrap();

        assert_eq!(result.encoder_input.shape(), &[6, 2, 2, 2, 3]);
        assert_eq!(result.feature_token_counts, vec![6]);
        assert_eq!(result.item_sizes, vec![(4, 4)]);
    }

    #[test]
    fn scaled_dimensions_match_expected_rounding_and_cap_rule() {
        let processor = InklingImageProcessor {
            patch_size: DEFAULT_PATCH_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            rescale_image_frac: Some(1.5),
            rescale_image_max_upscaled_long_edge: None,
            max_patches: None,
            mean: INKLING_IMAGE_MEAN,
            std: INKLING_IMAGE_STD,
        };

        assert_eq!(processor.scaled_image_dimensions(3, 2), (5, 3));

        let capped_processor = InklingImageProcessor {
            patch_size: DEFAULT_PATCH_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            rescale_image_frac: Some(2.0),
            rescale_image_max_upscaled_long_edge: Some(
                DEFAULT_RESCALE_IMAGE_MAX_UPSCALED_LONG_EDGE,
            ),
            max_patches: None,
            mean: INKLING_IMAGE_MEAN,
            std: INKLING_IMAGE_STD,
        };
        assert_eq!(
            capped_processor.scaled_image_dimensions(1200, 600),
            (2048, 1024)
        );
        assert_eq!(
            capped_processor.scaled_image_dimensions(3000, 2000),
            (3000, 2000)
        );
    }

    #[test]
    fn inkling_wire_is_the_float_tensor_written_as_bfloat16_where_the_norm_names_it() {
        let mut wide = RgbImage::new(83, 121);
        for (x, y, pixel) in wide.enumerate_pixels_mut() {
            *pixel = Rgb([(x * 3) as u8, (y * 2) as u8, (x + y) as u8]);
        }
        let (wide, small) = (DynamicImage::ImageRgb8(wide), test_image());
        let images = [&wide, &small];
        let processor = processor_without_rescale();
        let two_pass = processor.preprocess(&images).unwrap();
        let wire = processor
            .preprocess_wire(&images, EncoderDtype::BFloat16)
            .unwrap()
            .unwrap();
        assert_eq!(wire.feature_token_counts, two_pass.feature_token_counts);
        assert_eq!(wire.encoder_input.shape(), two_pass.encoder_input.shape());
        assert!(
            wire.encoder_input
                .into_bytes_as(EncoderDtype::BFloat16)
                .unwrap()
                == two_pass.encoder_input.write_as(EncoderDtype::BFloat16),
            "the wire differs from the float tensor's bytes"
        );
        for dtype in [EncoderDtype::Float16, EncoderDtype::Float32] {
            assert!(processor.preprocess_wire(&images, dtype).is_none());
        }
        // With a mean and a std of one half, three levels of byte 127 narrow
        // to another bfloat16 than the byte-unit norm names.
        let halves = InklingImageProcessor {
            mean: [0.5; 3],
            std: [0.5; 3],
            ..processor_without_rescale()
        };
        assert!(halves
            .preprocess_wire(&images, EncoderDtype::BFloat16)
            .is_none());
    }
}
