//! Gemma 4 image and video preprocessing.
//!
//! Port of transformers' `Gemma4ImageProcessor` (its torchvision backend, the
//! one the serving engine runs) in the form the engine's Gemma 4 model consumes: per image a
//! flat `(max_patches, patch_pixels)` tensor of `patch_size` x `patch_size`
//! RGB patches in row-major order, zero-padded to `max_patches =
//! max_soft_tokens * pooling_kernel_size^2`, and per patch its `(x, y)` grid
//! coordinates, `-1` for the padding (`pixel_position_ids` on the engine
//! side). The image is resized with its aspect ratio kept to the largest size
//! whose sides are multiples of `pooling_kernel_size * patch_size` within the
//! patch budget (the reference `get_aspect_ratio_preserving_size`), rescaled
//! by `rescale_factor` and not normalized (the encoder standardizes on device).
//! The vision tower pools every `pooling_kernel_size`^2 block of patches into
//! one soft token, so an image costs `num_patches / pooling_kernel_size^2`
//! tokens: 280 at most with the default budget, 256 for a square image.
//!
//! Video has no processor of its own in the engine: it runs the image
//! processor over the sampled frames with the video budget (70 soft tokens per
//! frame by default), so a clip is encoded here as its frames stacked along
//! the first dimension, with `video_frame_counts` saying how many belong to
//! each clip.

use std::borrow::Cow;

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array3;

use crate::{
    encoder_inputs::ModelSpecificValue,
    types::{Modality, RgbFrameRef},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VideoClipOptions, VisionPreProcessor},
        transforms::{
            pil_to_filter, resize_bicubic_torch_aa_rgb, resize_rgb_bytes, rgb_bytes, TransformError,
        },
    },
};

const DEFAULT_PATCH_SIZE: usize = 16;
const DEFAULT_POOLING_KERNEL_SIZE: usize = 3;
const DEFAULT_IMAGE_SOFT_TOKENS: usize = 280;
/// The per-frame budget the engine applies to video, the reference video
/// processor's default; the engine does not read it from the config.
const DEFAULT_VIDEO_SOFT_TOKENS: usize = 70;
/// The budgets the reference processor accepts (`_SUPPORTED_SOFT_TOKENS`).
const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];
const CHANNELS: usize = 3;

/// The geometry a config pins: patch size, pooling kernel and token budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Params {
    patch: usize,
    pool: usize,
    max_soft_tokens: usize,
}

impl Params {
    fn from_config(config: &PreProcessorConfig, video: bool) -> Result<Self, TransformError> {
        let patch = config.get_patch_size(DEFAULT_PATCH_SIZE);
        let pool = extra_usize(config, "pooling_kernel_size", DEFAULT_POOLING_KERNEL_SIZE)?;
        // The engine feeds frames to the image processor at the video processor's
        // default budget whatever the configs say; the video processor's own
        // config is honoured when it is the one handed in (its budget is that
        // default on the released checkpoints), the image config's budget is
        // not applied to frames.
        let max_soft_tokens = if video && !config.extra.contains_key("video_processor_type") {
            DEFAULT_VIDEO_SOFT_TOKENS
        } else {
            let default = if video {
                DEFAULT_VIDEO_SOFT_TOKENS
            } else {
                DEFAULT_IMAGE_SOFT_TOKENS
            };
            extra_usize(config, "max_soft_tokens", default)?
        };
        if patch == 0 || pool == 0 {
            return Err(shape(
                "Gemma 4 patch_size and pooling_kernel_size must be positive",
            ));
        }
        if !SUPPORTED_SOFT_TOKENS.contains(&max_soft_tokens) {
            return Err(shape(format!(
                "Gemma 4 max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS:?}, got {max_soft_tokens}"
            )));
        }
        Ok(Self {
            patch,
            pool,
            max_soft_tokens,
        })
    }

    fn max_patches(self) -> usize {
        self.max_soft_tokens * self.pool * self.pool
    }

    /// The multiple each resized side is rounded down to.
    fn side_mult(self) -> usize {
        self.pool * self.patch
    }

    fn patch_pixels(self) -> usize {
        CHANNELS * self.patch * self.patch
    }

    /// The reference `get_aspect_ratio_preserving_size`: the `(height, width)`
    /// an image of this size is resized to. The largest size that keeps the
    /// aspect ratio, fits `max_patches` patches and has both sides divisible
    /// by `pooling_kernel_size * patch_size`; a side that rounds to nothing
    /// becomes one block and the other side the aspect ratio's worth, within
    /// the budget.
    fn target_size(self, height: usize, width: usize) -> Result<(usize, usize), TransformError> {
        if height == 0 || width == 0 {
            return Err(shape(format!(
                "Gemma 4 cannot resize an empty {width}x{height} image"
            )));
        }
        let side = self.side_mult();
        let target_px = (self.max_patches() * self.patch * self.patch) as f64;
        let factor = (target_px / (height * width) as f64).sqrt();
        let round_down = |ideal: f64| (ideal / side as f64).floor() as usize * side;
        let mut target_h = round_down(factor * height as f64);
        let mut target_w = round_down(factor * width as f64);
        if target_h == 0 && target_w == 0 {
            return Err(shape(format!(
                "Gemma 4 cannot resize {width}x{height} to a multiple of {side} pixels"
            )));
        }
        let max_side = (self.max_patches() / (self.pool * self.pool)) * side;
        if target_h == 0 {
            target_h = side;
            target_w = ((width / height) * side).min(max_side);
        } else if target_w == 0 {
            target_w = side;
            target_h = ((height / width) * side).min(max_side);
        }
        if (target_h * target_w) as f64 > target_px {
            return Err(shape(format!(
                "Gemma 4 resize of {width}x{height} to {target_w}x{target_h} exceeds {} patches",
                self.max_patches()
            )));
        }
        Ok((target_h, target_w))
    }

    /// Soft tokens for a resized `(height, width)`: the pooled patch grid.
    fn soft_tokens(self, target: (usize, usize)) -> usize {
        (target.0 / self.patch) * (target.1 / self.patch) / (self.pool * self.pool)
    }
}

/// The pixel transform as a per-channel table: `rescale_factor` times the
/// byte, and the fused normalization of the reference when it is not the
/// identity. An identity normalization (mean 0, std 1) is skipped: the
/// engine's image processor never normalizes, and `x * rescale` is its
/// arithmetic where `(x - 0) / (1 / rescale)` could differ in the last bit.
fn pixel_lut(config: &PreProcessorConfig) -> Result<[[f32; 256]; 3], TransformError> {
    let rescale = if config.do_rescale.unwrap_or(true) {
        config.rescale_factor.unwrap_or(1.0 / 255.0)
    } else {
        1.0
    };
    let mean = config.image_mean.clone().unwrap_or_else(|| vec![0.0; 3]);
    let std = config.image_std.clone().unwrap_or_else(|| vec![1.0; 3]);
    let identity = mean.iter().all(|value| *value == 0.0) && std.iter().all(|value| *value == 1.0);
    let normalize = config.do_normalize.unwrap_or(false) && !identity;
    if mean.len() < CHANNELS || std.len() < CHANNELS {
        return Err(shape(
            "Gemma 4 image_mean and image_std need three channels",
        ));
    }
    if !rescale.is_finite()
        || (normalize
            && (mean.iter().any(|value| !value.is_finite())
                || std.iter().any(|value| !value.is_finite() || *value == 0.0)))
    {
        return Err(shape(
            "invalid Gemma 4 pixel rescale or normalization values",
        ));
    }
    let lut = std::array::from_fn(|channel| {
        if normalize {
            // The reference fuses the rescale into the normalization:
            // `(x - mean / rescale) / (std / rescale)`, in float32.
            let fused_mean = mean[channel] as f32 * (1.0 / rescale) as f32;
            let fused_std = std[channel] as f32 * (1.0 / rescale) as f32;
            std::array::from_fn(|value| (value as f32 - fused_mean) / fused_std)
        } else {
            let scale = rescale as f32;
            std::array::from_fn(|value| value as f32 * scale)
        }
    });
    lut.iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(lut)
        .ok_or_else(|| shape("Gemma 4 pixel transform overflow"))
}

/// The pixel pipeline a config sets: whether an image is resized, the
/// filter it is resized with, and the per-channel table of [`pixel_lut`].
#[derive(Debug, Clone)]
struct Pixels {
    do_resize: bool,
    filter: FilterType,
    lut: [[f32; 256]; 3],
}

impl Pixels {
    fn from_config(config: &PreProcessorConfig) -> Result<Self, TransformError> {
        Ok(Self {
            do_resize: config.do_resize.unwrap_or(true),
            filter: pil_to_filter(config.resampling.or(Some(3))),
            lut: pixel_lut(config)?,
        })
    }
}

trait RgbSource {
    fn dimensions(&self) -> (u32, u32);
    fn rgb(&self) -> Result<Cow<'_, [u8]>, TransformError>;
}

impl RgbSource for DynamicImage {
    fn dimensions(&self) -> (u32, u32) {
        GenericImageView::dimensions(self)
    }

    fn rgb(&self) -> Result<Cow<'_, [u8]>, TransformError> {
        let (_, _, data) = rgb_bytes(self);
        Ok(data)
    }
}

impl RgbSource for RgbFrameRef<'_> {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn rgb(&self) -> Result<Cow<'_, [u8]>, TransformError> {
        let expected = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(CHANNELS);
        if self.data.len() != expected {
            return Err(shape(format!(
                "Gemma 4 RGB frame has {} bytes, expected {expected} for {}x{}",
                self.data.len(),
                self.width,
                self.height
            )));
        }
        Ok(Cow::Borrowed(self.data))
    }
}

/// One encoded image or frame: `max_patches * patch_pixels` values,
/// `max_patches * 2` positions and the soft-token count.
struct Encoded {
    values: Vec<f32>,
    positions: Vec<i64>,
    tokens: usize,
}

/// The resized, patchified and padded form of one image or frame.
fn encode<F: RgbSource>(
    frame: &F,
    params: Params,
    pixels: &Pixels,
) -> Result<Encoded, TransformError> {
    let (source_w, source_h) = frame.dimensions();
    let (source_w, source_h) = (source_w as usize, source_h as usize);
    let raw = frame.rgb()?;
    let lut = &pixels.lut;
    let (height, width) = if pixels.do_resize {
        params.target_size(source_h, source_w)?
    } else {
        (source_h, source_w)
    };
    let side = params.side_mult();
    if height == 0 || width == 0 || !height.is_multiple_of(side) || !width.is_multiple_of(side) {
        return Err(shape(format!(
            "Gemma 4 input {width}x{height} is not a multiple of {side} pixels"
        )));
    }
    let (grid_h, grid_w) = (height / params.patch, width / params.patch);
    let num_patches = grid_h * grid_w;
    if num_patches > params.max_patches() {
        return Err(shape(format!(
            "Gemma 4 input {width}x{height} has {num_patches} patches, above the budget of {}",
            params.max_patches()
        )));
    }
    let pixels: Cow<'_, [u8]> = if (height, width) == (source_h, source_w) {
        raw
    } else {
        let filter = pixels.filter;
        let resized = if filter == FilterType::CatmullRom {
            resize_bicubic_torch_aa_rgb(
                raw.as_ref(),
                source_w as u32,
                source_h as u32,
                width as u32,
                height as u32,
            )?
        } else {
            resize_rgb_bytes(
                raw.as_ref(),
                source_w as u32,
                source_h as u32,
                width as u32,
                height as u32,
                filter,
            )?
        };
        Cow::Owned(resized.into_raw())
    };

    let patch_pixels = params.patch_pixels();
    let mut values = vec![0.0_f32; params.max_patches() * patch_pixels];
    let row_bytes = width * CHANNELS;
    for patch_y in 0..grid_h {
        for patch_x in 0..grid_w {
            // Patch elements run `[dy][dx][channel]`, the interleaved layout
            // of the source rows (the reference's `(ph, pw, p, p, c)` view).
            let patch = &mut values[(patch_y * grid_w + patch_x) * patch_pixels..][..patch_pixels];
            for dy in 0..params.patch {
                let source =
                    (patch_y * params.patch + dy) * row_bytes + patch_x * params.patch * CHANNELS;
                let row = &pixels[source..source + params.patch * CHANNELS];
                let out = &mut patch[dy * params.patch * CHANNELS..][..params.patch * CHANNELS];
                for (out, (value, lut)) in out.iter_mut().zip(row.iter().zip(lut.iter().cycle())) {
                    *out = lut[*value as usize];
                }
            }
        }
    }
    // `(x, y)` per patch in reading order, `-1` for the padding.
    let mut positions = vec![-1_i64; params.max_patches() * 2];
    for (index, position) in positions[..num_patches * 2]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .enumerate()
    {
        position[0] = (index % grid_w) as i64;
        position[1] = (index / grid_w) as i64;
    }
    Ok(Encoded {
        values,
        positions,
        tokens: num_patches / (params.pool * params.pool),
    })
}

/// Stack encoded items along the first dimension.
fn stack(
    encoded: Vec<Encoded>,
    params: Params,
) -> Result<(Array3<f32>, ModelSpecificValue, Vec<usize>), TransformError> {
    let count = encoded.len();
    let (max_patches, patch_pixels) = (params.max_patches(), params.patch_pixels());
    let mut values = Vec::new();
    values
        .try_reserve_exact(count * max_patches * patch_pixels)
        .map_err(|error| shape(format!("cannot reserve Gemma 4 patch output: {error}")))?;
    let mut positions = Vec::with_capacity(count * max_patches * 2);
    let mut tokens = Vec::with_capacity(count);
    for item in encoded {
        values.extend_from_slice(&item.values);
        positions.extend_from_slice(&item.positions);
        tokens.push(item.tokens);
    }
    let tensor = Array3::from_shape_vec((count, max_patches, patch_pixels), values)
        .map_err(|error| shape(format!("failed to build Gemma 4 patches: {error}")))?;
    let positions = ModelSpecificValue::IntTensor {
        data: positions,
        shape: vec![count, max_patches, 2],
    };
    Ok((tensor, positions, tokens))
}

fn preprocess_video_frames<F: RgbSource>(
    frames: &[F],
    params: Params,
    pixels: &Pixels,
) -> Result<PreprocessedEncoderInputs, TransformError> {
    let first = frames.first().ok_or(TransformError::EmptyBatch)?;
    let (width, height) = first.dimensions();
    if frames
        .iter()
        .any(|frame| frame.dimensions() != (width, height))
    {
        return Err(shape("Gemma 4 video frames must have identical dimensions"));
    }
    let encoded = frames
        .iter()
        .map(|frame| encode(frame, params, pixels))
        .collect::<Result<Vec<_>, _>>()?;
    let (tensor, positions, tokens) = stack(encoded, params)?;
    Ok(
        PreprocessedEncoderInputs::new(tensor, vec![tokens.iter().sum()], vec![(width, height)])
            .with_extra("pixel_position_ids_videos", positions)
            .with_extra(
                "video_frame_counts",
                ModelSpecificValue::int_1d(vec![frames.len() as i64]),
            ),
    )
}

/// A `usize` extra key under [`PreProcessorConfig::extra_usize`]'s rule, absent -> `default`.
fn extra_usize(
    config: &PreProcessorConfig,
    key: &str,
    default: usize,
) -> Result<usize, TransformError> {
    config
        .extra_usize(key)
        .map(|value| value.unwrap_or(default))
        .map_err(|value| {
            shape(format!(
                "Gemma 4 preprocessor key {key} must be a non-negative integer, got {value}"
            ))
        })
}

fn shape(message: impl Into<String>) -> TransformError {
    TransformError::ShapeError(message.into())
}

/// Gemma 4 image and video preprocessor.
#[derive(Debug, Clone)]
pub struct Gemma4Processor {
    /// The modality the processor was built for, whose geometry `params` is.
    modality: Modality,
    params: Params,
    pixels: Pixels,
}

impl Default for Gemma4Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl Gemma4Processor {
    /// The image processor of the default preprocessor config.
    #[expect(
        clippy::expect_used,
        reason = "the default config pins a supported geometry and an identity normalization"
    )]
    pub fn new() -> Self {
        Self::from_config_for(&PreProcessorConfig::default(), Modality::Image)
            .expect("the default preprocessor config is valid")
    }

    /// Resolve the geometry of `modality` (image or video) and the pixel
    /// transform of `config` once; a config the processor cannot serve for
    /// that modality is refused here. Only that modality's settings are
    /// read, so a setting only the other modality's path would refuse does
    /// not stop it. Any other modality is refused.
    pub fn from_config_for(
        config: &PreProcessorConfig,
        modality: Modality,
    ) -> Result<Self, TransformError> {
        let video = match modality {
            Modality::Image => false,
            Modality::Video => true,
            other => {
                return Err(shape(format!(
                    "Gemma 4 preprocesses images and video, not {other}"
                )))
            }
        };
        let params = Params::from_config(config, video)?;
        Ok(Self {
            modality,
            params,
            pixels: Pixels::from_config(config)?,
        })
    }

    /// The geometry for a call of `modality`: this processor's own, or the
    /// error for a call of the modality it was not built for.
    fn params_for(&self, modality: Modality) -> Result<Params, TransformError> {
        if modality == self.modality {
            Ok(self.params)
        } else {
            Err(shape(format!(
                "this Gemma 4 processor was built for {} preprocessing, not {modality}",
                self.modality
            )))
        }
    }
}

impl VisionPreProcessor for Gemma4Processor {
    /// Each image is resized, patchified and padded on its own.
    fn supports_per_image_preprocessing(&self) -> bool {
        true
    }

    fn default_mean(&self) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }

    fn default_std(&self) -> [f64; 3] {
        [1.0, 1.0, 1.0]
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let params = self.params_for(Modality::Image)?;
        let encoded = images
            .iter()
            .map(|image| encode(image, params, &self.pixels))
            .collect::<Result<Vec<_>, _>>()?;
        let (tensor, positions, tokens) = stack(encoded, params)?;
        let sizes = images.iter().map(GenericImageView::dimensions).collect();
        Ok(PreprocessedEncoderInputs::new(tensor, tokens, sizes)
            .with_extra("pixel_position_ids", positions))
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        _clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        preprocess_video_frames(frames, self.params_for(Modality::Video)?, &self.pixels)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        _clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        preprocess_video_frames(frames, self.params_for(Modality::Video)?, &self.pixels)
    }

    /// Soft tokens for an image of this size; the preprocessing error for a
    /// size the reference rule cannot resize.
    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let params = self.params_for(Modality::Image)?;
        let target = params.target_size(height as usize, width as usize)?;
        Ok(params.soft_tokens(target))
    }

    fn model_name(&self) -> &'static str {
        "gemma4"
    }

    /// The size depends on the image's aspect ratio.
    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    const IMAGE_CONFIG: &str = r#"{
        "do_convert_rgb": true, "do_normalize": false, "do_rescale": true, "do_resize": true,
        "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
        "image_processor_type": "Gemma4ImageProcessor", "image_seq_length": 280,
        "max_soft_tokens": 280, "patch_size": 16, "pooling_kernel_size": 3, "resample": 3,
        "rescale_factor": 0.00392156862745098
    }"#;

    const VIDEO_CONFIG: &str = r#"{
        "do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "do_resize": true,
        "do_sample_frames": true, "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
        "max_soft_tokens": 70, "num_frames": 32, "patch_size": 16, "pooling_kernel_size": 3,
        "resample": 3, "rescale_factor": 0.00392156862745098, "return_metadata": false,
        "video_processor_type": "Gemma4VideoProcessor"
    }"#;

    fn image_config() -> PreProcessorConfig {
        PreProcessorConfig::from_json(IMAGE_CONFIG).unwrap()
    }

    fn video_config() -> PreProcessorConfig {
        PreProcessorConfig::from_json(VIDEO_CONFIG).unwrap()
    }

    fn params() -> Params {
        Params::from_config(&image_config(), false).unwrap()
    }

    fn solid(width: u32, height: u32, color: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb(color)))
    }

    /// Target sizes and token counts of the reference `get_aspect_ratio_preserving_size`
    /// (height, width in; height, width out), as transformers computes them.
    /// A processor is built for images or video; any other modality is
    /// refused when it is built rather than on every call.
    #[test]
    fn construction_refuses_a_modality_it_does_not_preprocess() {
        for modality in [Modality::Audio, Modality::ImageEmbeds] {
            assert!(
                Gemma4Processor::from_config_for(&PreProcessorConfig::default(), modality).is_err(),
                "{modality}"
            );
        }
    }

    #[test]
    fn target_sizes_follow_the_reference_rule() {
        let params = params();
        let cases = [
            ((512, 512), (768, 768), 256),
            ((64, 64), (768, 768), 256),
            ((1080, 1920), (576, 1056), 264),
            ((300, 1200), (384, 1584), 264),
            ((1200, 300), (1584, 384), 264),
            ((128, 96), (912, 672), 266),
            ((2000, 3000), (624, 960), 260),
            ((3, 2), (960, 624), 260),
            ((1, 1), (768, 768), 256),
            // The zero-side rule: one block high, the aspect ratio wide, capped at the budget.
            ((10, 3000), (48, 13440), 280),
            ((3000, 10), (13440, 48), 280),
            ((20, 3000), (48, 9792), 204),
        ];
        for ((h, w), (th, tw), tokens) in cases {
            let target = params.target_size(h, w).unwrap();
            assert_eq!(target, (th, tw), "{w}x{h}");
            assert_eq!(params.soft_tokens(target), tokens, "{w}x{h}");
        }
        assert_eq!(params.max_patches(), 2520);
        assert_eq!(params.side_mult(), 48);
        assert!(params.target_size(0, 10).is_err());
    }

    #[test]
    fn video_frames_take_the_video_budget() {
        let video = Params::from_config(&video_config(), true).unwrap();
        assert_eq!(video.max_soft_tokens, 70);
        assert_eq!(video.max_patches(), 630);
        // 320x240 frames: 336x432 -> 21 x 27 patches -> 63 tokens.
        assert_eq!(video.target_size(240, 320).unwrap(), (336, 432));
        assert_eq!(video.soft_tokens((336, 432)), 63);
        // Handed the image config, frames keep the engine's video budget.
        let from_image = Params::from_config(&image_config(), true).unwrap();
        assert_eq!(from_image.max_soft_tokens, 70);
        // An unsupported budget is refused like the reference refuses it.
        let odd: PreProcessorConfig =
            serde_json::from_value(serde_json::json!({"max_soft_tokens": 100})).unwrap();
        assert!(Params::from_config(&odd, false).is_err());
    }

    #[test]
    fn preprocess_emits_the_engine_contract() {
        let image = solid(512, 512, [255, 0, 51]);
        let out = Gemma4Processor::from_config_for(&image_config(), Modality::Image)
            .unwrap()
            .preprocess(std::slice::from_ref(&image))
            .unwrap();
        assert_eq!(out.encoder_input.shape(), &[1, 2520, 768]);
        assert_eq!(out.feature_token_counts, vec![256]);
        assert_eq!(out.item_sizes, vec![(512, 512)]);
        let values = out.encoder_input.as_f32();
        // 768x768 -> 48 x 48 patches = 2304 real patches, then zero padding.
        let red = 255.0_f32 * (1.0_f32 / 255.0);
        let blue = 51.0_f32 * (1.0_f32 / 255.0);
        assert_eq!(values[[0, 0, 0]], red);
        assert_eq!(values[[0, 0, 1]], 0.0);
        assert_eq!(values[[0, 0, 2]], blue);
        assert_eq!(values[[0, 2303, 767]], blue);
        assert_eq!(values[[0, 2304, 0]], 0.0);
        assert_eq!(values[[0, 2519, 767]], 0.0);
        let ModelSpecificValue::IntTensor { data, shape } =
            &out.model_specific["pixel_position_ids"]
        else {
            panic!("pixel_position_ids must be an int tensor");
        };
        assert_eq!(shape, &[1, 2520, 2]);
        assert_eq!(&data[..6], &[0, 0, 1, 0, 2, 0]);
        assert_eq!(&data[96..100], &[0, 1, 1, 1]);
        assert_eq!(&data[2303 * 2..2304 * 2], &[47, 47]);
        assert!(data[2304 * 2..].iter().all(|&v| v == -1));
    }

    #[test]
    fn batches_stack_images_of_different_sizes() {
        let images = [solid(64, 64, [1, 2, 3]), solid(1920, 1080, [4, 5, 6])];
        let out = Gemma4Processor::from_config_for(&image_config(), Modality::Image)
            .unwrap()
            .preprocess(&images)
            .unwrap();
        assert_eq!(out.encoder_input.shape(), &[2, 2520, 768]);
        assert_eq!(out.feature_token_counts, vec![256, 264]);
        assert_eq!(out.item_sizes, vec![(64, 64), (1920, 1080)]);
        let values = out.encoder_input.as_f32();
        assert_eq!(values[[1, 0, 0]], 4.0 * (1.0_f32 / 255.0));
        // 1056x576 -> 66 x 36 = 2376 patches; the rest is padding.
        assert_eq!(values[[1, 2375, 0]], 4.0 * (1.0_f32 / 255.0));
        assert_eq!(values[[1, 2376, 0]], 0.0);
    }

    #[test]
    fn video_stacks_frames_and_counts_them() {
        let frames = vec![solid(320, 240, [9, 9, 9]); 3];
        let processor = Gemma4Processor::from_config_for(&video_config(), Modality::Video).unwrap();
        let clip = VideoClipOptions::sampled_at(2.0);
        let out = processor.preprocess_video(&frames, &clip).unwrap();
        assert_eq!(out.encoder_input.shape(), &[3, 630, 768]);
        assert_eq!(out.feature_token_counts, vec![3 * 63]);
        assert_eq!(out.item_sizes, vec![(320, 240)]);
        let ModelSpecificValue::IntTensor { data, shape } =
            &out.model_specific["pixel_position_ids_videos"]
        else {
            panic!("pixel_position_ids_videos must be an int tensor");
        };
        assert_eq!(shape, &[3, 630, 2]);
        // 432x336 -> 27 x 21 patches = 567 real, 63 padded, per frame.
        assert_eq!(&data[566 * 2..567 * 2], &[26, 20]);
        assert_eq!(&data[567 * 2..568 * 2], &[-1, -1]);
        assert_eq!(&data[630 * 2..630 * 2 + 2], &[0, 0]);
        let ModelSpecificValue::IntTensor { data, .. } = &out.model_specific["video_frame_counts"]
        else {
            panic!("video_frame_counts must be an int tensor");
        };
        assert_eq!(data, &[3]);

        // The borrowed-frame path is the same computation.
        let rgb: Vec<Vec<u8>> = frames.iter().map(|f| f.to_rgb8().into_raw()).collect();
        let refs: Vec<RgbFrameRef<'_>> = rgb
            .iter()
            .map(|data| RgbFrameRef {
                width: 320,
                height: 240,
                data,
            })
            .collect();
        let borrowed = processor.preprocess_video_rgb(&refs, &clip).unwrap();
        assert_eq!(
            borrowed.encoder_input.as_f32().as_slice(),
            out.encoder_input.as_f32().as_slice()
        );
        assert_eq!(borrowed.feature_token_counts, out.feature_token_counts);

        let mixed = [solid(320, 240, [0; 3]), solid(321, 240, [0; 3])];
        assert!(processor.preprocess_video(&mixed, &clip).is_err());
        assert!(processor.preprocess_video(&[], &clip).is_err());
    }

    #[test]
    fn construction_refuses_a_config_the_parameters_cannot_read() {
        // A config the parameters cannot be read from is the parameters' error
        // when the processor is built, not a count of zero later.
        let mut config = image_config();
        config.extra.insert(
            "pooling_kernel_size".to_string(),
            serde_json::json!("three"),
        );
        let constructed = Gemma4Processor::from_config_for(&config, Modality::Image)
            .err()
            .unwrap()
            .to_string();
        let read = Params::from_config(&config, false).unwrap_err().to_string();
        assert_eq!(constructed, read);
    }

    #[test]
    fn token_counts_and_degenerate_inputs() {
        let processor = Gemma4Processor::from_config_for(&image_config(), Modality::Image).unwrap();
        assert_eq!(processor.calculate_num_tokens(512, 512).unwrap(), 256);
        assert_eq!(processor.calculate_num_tokens(1920, 1080).unwrap(), 264);
        assert_eq!(processor.calculate_num_tokens(3000, 10).unwrap(), 280);
        assert!(processor.calculate_num_tokens(0, 10).is_err());
        assert_eq!(processor.model_name(), "gemma4");
        assert!(processor.supports_per_image_preprocessing());
        assert_eq!(processor.get_processed_size(), None);
        assert!(processor.preprocess(&[]).is_err());
        // No resize: the input must already be block-aligned and within budget.
        let no_resize: PreProcessorConfig =
            serde_json::from_value(serde_json::json!({"do_resize": false})).unwrap();
        let no_resize = Gemma4Processor::from_config_for(&no_resize, Modality::Image).unwrap();
        assert!(no_resize.preprocess(&[solid(100, 100, [0; 3])]).is_err());
        let aligned = no_resize.preprocess(&[solid(96, 48, [0; 3])]).unwrap();
        assert_eq!(aligned.feature_token_counts, vec![2]);
        // Aligned but over the budget of 2520 patches (180 x 60 = 10,800).
        assert!(no_resize
            .preprocess(&[solid(48 * 60, 48 * 20, [0; 3])])
            .is_err());
    }

    #[test]
    fn non_identity_normalization_is_fused_like_the_reference() {
        let config: PreProcessorConfig = serde_json::from_value(serde_json::json!({
            "do_normalize": true, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
        }))
        .unwrap();
        let lut = pixel_lut(&config).unwrap();
        assert_eq!(lut[0][255], (255.0_f32 - 127.5) / 127.5);
        assert_eq!(lut[1][0], -1.0);
        let identity = pixel_lut(&video_config()).unwrap();
        assert_eq!(identity[2][255], 255.0_f32 * (1.0_f32 / 255.0));
        let bad: PreProcessorConfig = serde_json::from_value(
            serde_json::json!({"do_normalize": true, "image_mean": [0.0, 0.0, 0.0], "image_std": [0.0, 1.0, 1.0]}),
        )
        .unwrap();
        assert!(pixel_lut(&bad).is_err());
    }
}
