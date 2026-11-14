//! GLM-5.3-Flash image and video preprocessing.

use std::borrow::Cow;

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array2;

use super::qwen_vl_base::pixel_norm;
use crate::{
    encoder_inputs::{EncoderDtype, EncoderInput, ModelSpecificValue, PixelNorm},
    types::{Modality, RgbFrameRef},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VideoClipOptions, VisionPreProcessor},
        transforms::{
            pil_to_filter, resize_bicubic_torch_aa_rgb, resize_rgb_bytes, rgb_bytes,
            round_half_to_even, TransformError,
        },
    },
};

const PATCH_SIZE: usize = 14;
const MERGE_SIZE: usize = 2;
const TEMPORAL_PATCH_SIZE: usize = 2;
const MIN_IMAGE_TOKENS: usize = 16;
const MAX_IMAGE_TOKENS: usize = 8000;
/// The checkpoint's own video budget, in tokens.
const MAX_VIDEO_TOKENS: usize = 240_000;

#[derive(Clone, Copy)]
struct Params {
    patch: usize,
    merge: usize,
    temporal: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
    mean: [f64; 3],
    std: [f64; 3],
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    target: (usize, usize),
    content: (usize, usize),
}

/// The image's own bytes in the patch layout, with the normalization a
/// float destination applies.
struct Patches {
    values: Vec<u8>,
    norm: PixelNorm,
    grid: (usize, usize, usize),
    count: usize,
    tokens: usize,
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
        validate_rgb(self.width as usize, self.height as usize, self.data.len())?;
        Ok(Cow::Borrowed(self.data))
    }
}

/// A `usize` extra key under [`PreProcessorConfig::extra_usize`]'s rule (absent -> `default`,
/// a present-yet-unusable value a config error, since a silent default would change the
/// alignment factor and the token geometry).
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
                "GLM preprocessor key {key} must be a non-negative integer, got {value}"
            ))
        })
}

impl Params {
    fn from_config(config: &PreProcessorConfig, video: bool) -> Result<Self, TransformError> {
        let patch = config.get_patch_size(PATCH_SIZE);
        let merge = config.merge_size.unwrap_or(MERGE_SIZE);
        let temporal = config.temporal_patch_size.unwrap_or(TEMPORAL_PATCH_SIZE);
        let expand = extra_usize(config, "patch_expand_factor", 1)?;
        if [patch, merge, temporal, expand].contains(&0) {
            return Err(shape(
                "GLM patch, merge, temporal, and expand factors must be positive",
            ));
        }
        let factor = product(&[patch, merge, expand], "GLM alignment factor")?;
        let pixels_per_token = product(&[factor, factor, temporal], "GLM pixels per token")?;
        let budget = |pixels: Option<usize>, key: &str, fallback| {
            pixels.map_or_else(
                || {
                    product(
                        &[extra_usize(config, key, fallback)?, pixels_per_token],
                        "GLM pixel budget",
                    )
                },
                Ok,
            )
        };
        // Video budgets read video-named token keys first, then the video
        // processor config's `max_image_tokens`. An image config used as a
        // fallback keeps the checkpoint video default: its image-scale
        // `max_image_tokens` must not silently become a clip's volume cap.
        let (min_key, max_key, max_fallback) = if video {
            (
                "min_video_tokens",
                "max_video_tokens",
                video_token_budget(config)?,
            )
        } else {
            ("min_image_tokens", "max_image_tokens", MAX_IMAGE_TOKENS)
        };
        let min_pixels = budget(config.min_pixels, min_key, MIN_IMAGE_TOKENS)?;
        let max_pixels = budget(config.max_pixels, max_key, max_fallback)?;
        if min_pixels == 0 || min_pixels > max_pixels {
            return Err(shape("GLM pixel budget must be positive with min <= max"));
        }
        let triple = |values: &Option<Vec<f64>>, default| {
            values
                .as_ref()
                .filter(|values| values.len() >= 3)
                .map(|values| [values[0], values[1], values[2]])
                .unwrap_or(default)
        };
        Ok(Self {
            patch,
            merge,
            temporal,
            factor,
            min_pixels,
            max_pixels,
            mean: triple(&config.image_mean, PreProcessorConfig::CLIP_MEAN),
            std: triple(&config.image_std, PreProcessorConfig::CLIP_STD),
        })
    }

    /// The geometry the encode works on for `frames` frames of `height` x
    /// `width`: the budget search's target when `do_resize` is on, the source
    /// itself otherwise, refused as the encode refuses it (a zero side, a
    /// canvas off the factor). The count and the encode take their canvas
    /// from here, so they agree in both modes.
    fn input_geometry(
        self,
        do_resize: bool,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Geometry, TransformError> {
        let geometry = if do_resize {
            self.geometry(frames, height, width)?
        } else {
            if height == 0 || width == 0 {
                return Err(shape("GLM dimensions must be positive"));
            }
            Geometry {
                target: (height, width),
                content: (height, width),
            }
        };
        let (target_h, target_w) = geometry.target;
        if !target_h.is_multiple_of(self.factor) || !target_w.is_multiple_of(self.factor) {
            return Err(shape("GLM canvas is not aligned to the configured factor"));
        }
        Ok(geometry)
    }

    fn geometry(
        self,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Geometry, TransformError> {
        if frames == 0 || height == 0 || width == 0 {
            return Err(shape("GLM dimensions must be positive"));
        }
        // Matches the reference family's guard (see qwen smart_resize):
        // beyond 200:1 the budget search degenerates to a 1-pixel strip on a
        // (factor, factor) canvas, silently destroying the image.
        let (long, short) = (height.max(width) as f64, height.min(width) as f64);
        if long / short > 200.0 {
            return Err(shape(format!(
                "GLM aspect ratio must be below 200:1, got {height}x{width}"
            )));
        }
        let align =
            |value: usize| product(&[value.div_ceil(self.factor), self.factor], "GLM alignment");
        let aligned_frames = (round_half_to_even(frames as f64 / self.temporal as f64) as usize
            * self.temporal)
            .max(self.temporal);
        let volume = |n: usize, h: usize, w: usize| {
            (n as u128)
                .saturating_mul(h as u128)
                .saturating_mul(w as u128)
        };
        let mut target = (align(height)?, align(width)?);
        let mut target_volume = volume(aligned_frames, target.0, target.1);

        if target_volume < self.min_pixels as u128 {
            let scale =
                (self.min_pixels as f64 / (frames as f64 * height as f64 * width as f64)).sqrt();
            target = (
                align(((height as f64 * scale).ceil() as usize).max(1))?,
                align(((width as f64 * scale).ceil() as usize).max(1))?,
            );
            target_volume = volume(aligned_frames, target.0, target.1);
        }

        if target_volume > self.max_pixels as u128 {
            if volume(aligned_frames, self.factor, self.factor) > self.max_pixels as u128 {
                return Err(shape("GLM max_pixels is smaller than one aligned patch"));
            }
            let (mut low, mut high, mut best) = (1usize, height, (self.factor, self.factor));
            while low <= high {
                let candidate_h = low + (high - low) / 2;
                let candidate_w =
                    ((width as u128 * candidate_h as u128) / height as u128).max(1) as usize;
                let candidate = (align(candidate_h)?, align(candidate_w)?);
                if volume(aligned_frames, candidate.0, candidate.1) <= self.max_pixels as u128 {
                    best = candidate;
                    low = candidate_h + 1;
                } else {
                    high = candidate_h - 1;
                }
            }
            target = best;
        }

        let mut scale = (target.0 as f64 / height as f64).min(target.1 as f64 / width as f64);
        if volume(frames, height, width) >= self.min_pixels as u128 {
            scale = scale.min(1.0);
        }
        Ok(Geometry {
            target,
            content: (
                ((height as f64 * scale).floor() as usize).clamp(1, target.0),
                ((width as f64 * scale).floor() as usize).clamp(1, target.1),
            ),
        })
    }
}

/// A video processor config's `max_image_tokens`, or the checkpoint video
/// default when the key is absent or an image config is used as a fallback.
fn video_token_budget(config: &PreProcessorConfig) -> Result<usize, TransformError> {
    if !config.extra.contains_key("video_processor_type") {
        return Ok(MAX_VIDEO_TOKENS);
    }
    if !config.extra.contains_key("max_image_tokens") {
        return Ok(MAX_VIDEO_TOKENS);
    }
    extra_usize(config, "max_image_tokens", MAX_VIDEO_TOKENS)
}

/// The reference's fused rescale and normalize for these settings
/// (`(byte - mean / rescale) / (std / rescale)`, the rounding transformers'
/// processor and an engine that normalizes on device share), refused where
/// a setting is not finite, a std is zero, or a byte would map to a
/// non-finite value.
fn glm_pixel_norm(
    config: &PreProcessorConfig,
    params: Params,
) -> Result<PixelNorm, TransformError> {
    let do_rescale = config.do_rescale.unwrap_or(true);
    let do_normalize = config.do_normalize.unwrap_or(true);
    let rescale = if do_rescale {
        config.get_rescale_factor()
    } else {
        1.0
    };
    if !rescale.is_finite()
        || (do_rescale && rescale == 0.0)
        || (do_normalize
            && (params.mean.iter().any(|value| !value.is_finite())
                || params
                    .std
                    .iter()
                    .any(|value| !value.is_finite() || *value == 0.0)))
    {
        return Err(shape("invalid GLM pixel rescale or normalization values"));
    }
    let channel_block = product(
        &[params.temporal, params.patch, params.patch],
        "GLM channel block",
    )?;
    let norm = pixel_norm(config, params.mean, params.std, channel_block);
    norm.table_f32()
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(norm)
        .ok_or_else(|| shape("GLM pixel transform overflow"))
}

/// The pixel pipeline a config sets: whether frames are resized, the filter
/// they are resized with, and the normalization of [`glm_pixel_norm`].
#[derive(Clone)]
struct Pixels {
    do_resize: bool,
    filter: FilterType,
    norm: PixelNorm,
}

impl Pixels {
    fn from_config(config: &PreProcessorConfig, params: Params) -> Result<Self, TransformError> {
        Ok(Self {
            do_resize: config.do_resize.unwrap_or(true),
            filter: pil_to_filter(config.resampling.or(Some(3))),
            norm: glm_pixel_norm(config, params)?,
        })
    }
}

/// One frame's bytes as the reference lays them out: resized to the content
/// size, then zero-padded to the target canvas. The reference image and video
/// processors resize with the torchvision antialiased kernel on the bytes
/// (a bicubic `resample` takes that kernel's uint8 form, which parts from
/// Pillow's quantization by up to two levels at the scale extremes: tiny
/// upscales and 4K-8K downscales), so a bicubic resize takes the same kernel here.
fn prepare<'a, F: RgbSource>(
    frame: &'a F,
    geometry: Geometry,
    filter: FilterType,
    do_resize: bool,
) -> Result<Cow<'a, [u8]>, TransformError> {
    let (source_w, source_h) = frame.dimensions();
    let raw = frame.rgb()?;
    let ((target_h, target_w), (content_h, content_w)) = (geometry.target, geometry.content);
    if !do_resize
        || (source_w as usize == target_w
            && source_h as usize == target_h
            && geometry.content == geometry.target)
    {
        return Ok(raw);
    }
    let content = if (source_w as usize, source_h as usize) == (content_w, content_h) {
        raw.into_owned()
    } else {
        let resized = if filter == FilterType::CatmullRom {
            resize_bicubic_torch_aa_rgb(
                raw.as_ref(),
                source_w,
                source_h,
                content_w as u32,
                content_h as u32,
            )?
        } else {
            resize_rgb_bytes(
                raw.as_ref(),
                source_w,
                source_h,
                content_w as u32,
                content_h as u32,
                filter,
            )?
        };
        resized.into_raw()
    };
    if geometry.content == geometry.target {
        return Ok(Cow::Owned(content));
    }
    let mut canvas = vec![0; rgb_len(target_w, target_h)?];
    let (content_row, target_row) = (content_w * 3, target_w * 3);
    for row in 0..content_h {
        canvas[row * target_row..row * target_row + content_row]
            .copy_from_slice(&content[row * content_row..(row + 1) * content_row]);
    }
    Ok(Cow::Owned(canvas))
}

/// Pack resized RGB frames into merged patch rows with temporal padding.
fn encode<F: RgbSource>(
    frames: &[F],
    budget_frames: usize,
    params: Params,
    pixels: &Pixels,
) -> Result<Patches, TransformError> {
    let first = frames.first().ok_or(TransformError::EmptyBatch)?;
    let source_size = first.dimensions();
    if frames.iter().any(|frame| frame.dimensions() != source_size) {
        return Err(shape("GLM video frames must have identical dimensions"));
    }
    let do_resize = pixels.do_resize;
    let geometry = params.input_geometry(
        do_resize,
        budget_frames,
        source_size.1 as usize,
        source_size.0 as usize,
    )?;
    let (height, width) = geometry.target;
    let (grid_h, grid_w) = (height / params.patch, width / params.patch);
    let grid_t = frames.len().div_ceil(params.temporal);
    let count = product(&[grid_t, grid_h, grid_w], "GLM patch count")?;
    let features = product(
        &[3, params.temporal, params.patch, params.patch],
        "GLM features",
    )?;
    let expected = product(&[count, features], "GLM patch output size")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(expected)
        .map_err(|error| shape(format!("cannot reserve GLM patch output: {error}")))?;
    // Write bounded row slices rather than growing the Vec for each pixel.
    // This also lets the gather loop check the RGB row's bounds once.
    values.resize(expected, 0);
    let mut remaining = values.as_mut_slice();
    let norm = pixels.norm;
    let filter = pixels.filter;
    let merged_patch = params.merge * params.patch;

    for temporal_index in 0..grid_t {
        let start = temporal_index * params.temporal;
        let end = (start + params.temporal).min(frames.len());
        let prepared = frames[start..end]
            .iter()
            .map(|frame| prepare(frame, geometry, filter, do_resize))
            .collect::<Result<Vec<_>, _>>()?;
        let last = prepared.last().ok_or(TransformError::EmptyBatch)?;
        for patch_row in 0..grid_h / params.merge {
            for patch_col in 0..grid_w / params.merge {
                let (y0, x0) = (patch_row * merged_patch, patch_col * merged_patch);
                for merge_h in 0..params.merge {
                    for merge_w in 0..params.merge {
                        for channel in 0..3 {
                            for temporal in 0..params.temporal {
                                let raw = prepared.get(temporal).unwrap_or(last).as_ref();
                                for patch_y in 0..params.patch {
                                    let row = (y0 + merge_h * params.patch + patch_y) * width
                                        + x0
                                        + merge_w * params.patch;
                                    let rgb = &raw[row * 3..(row + params.patch) * 3];
                                    let (output, rest) = remaining.split_at_mut(params.patch);
                                    remaining = rest;
                                    for (output, pixel) in
                                        output.iter_mut().zip(rgb.as_chunks::<3>().0)
                                    {
                                        *output = pixel[channel];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert!(remaining.is_empty());
    Ok(Patches {
        values,
        norm,
        grid: (grid_t, grid_h, grid_w),
        count,
        tokens: count / (params.merge * params.merge),
    })
}

fn preprocess_video_frames<F: RgbSource>(
    frames: &[F],
    params: Params,
    pixels: &Pixels,
    clip: &VideoClipOptions,
) -> Result<PreprocessedEncoderInputs, TransformError> {
    let original = frames
        .first()
        .ok_or(TransformError::EmptyBatch)?
        .dimensions();
    let patches = encode(frames, frames.len(), params, pixels)?;
    let fps = clip.sample_fps;
    if !fps.is_finite() || fps <= 0.0 {
        return Err(shape(format!("GLM video fps must be positive, got {fps}")));
    }
    let (grid_t, grid_h, grid_w) = patches.grid;
    let features = 3 * params.temporal * params.patch * params.patch;
    let tensor = Array2::from_shape_vec((patches.count, features), patches.values)
        .map_err(|error| shape(format!("failed to build GLM video patches: {error}")))?;
    Ok(PreprocessedEncoderInputs::new(
        EncoderInput::bytes(tensor, patches.norm),
        vec![patches.tokens],
        vec![original],
    )
    .with_extra(
        "video_grid_thw",
        ModelSpecificValue::int_2d(vec![grid_t as i64, grid_h as i64, grid_w as i64], 1, 3),
    )
    .with_extra(
        "patches_per_video",
        ModelSpecificValue::int_1d(vec![patches.count as i64]),
    )
    .with_extra(
        "patches_per_image",
        ModelSpecificValue::int_1d(vec![patches.count as i64]),
    )
    .with_extra(
        "video_second_per_grid",
        ModelSpecificValue::Tensor {
            data: vec![(params.temporal as f64 / fps) as f32],
            shape: vec![1],
        },
    ))
}

fn rgb_len(width: usize, height: usize) -> Result<usize, TransformError> {
    product(&[width, height, 3], "GLM RGB byte length")
}

fn product(values: &[usize], label: &str) -> Result<usize, TransformError> {
    values
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
        .ok_or_else(|| shape(format!("{label} overflow")))
}

fn validate_rgb(width: usize, height: usize, actual: usize) -> Result<(), TransformError> {
    let expected = rgb_len(width, height)?;
    if actual != expected {
        return Err(TransformError::InvalidShape {
            expected: format!("{width}x{height} RGB frame ({expected} bytes)"),
            actual: vec![actual],
        });
    }
    Ok(())
}

fn shape(message: impl Into<String>) -> TransformError {
    TransformError::ShapeError(message.into())
}

#[derive(Clone)]
pub struct Glm53FlashProcessor {
    /// The modality the processor was built for, whose geometry `params` is.
    modality: Modality,
    params: Params,
    pixels: Pixels,
}

impl Default for Glm53FlashProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Glm53FlashProcessor {
    /// The image processor of the default preprocessor config.
    #[expect(
        clippy::expect_used,
        reason = "the default config sets the checkpoint geometry and CLIP normalization"
    )]
    pub fn new() -> Self {
        Self::from_config_for(&PreProcessorConfig::default(), Modality::Image)
            .expect("the default preprocessor config is valid")
    }

    /// Resolve the budgets of `modality` (image or video) and the pixel
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
                    "GLM preprocesses images and video, not {other}"
                )))
            }
        };
        let params = Params::from_config(config, video)?;
        Ok(Self {
            modality,
            params,
            pixels: Pixels::from_config(config, params)?,
        })
    }

    /// The same padded canvas used by encoding, without allocating pixels.
    fn image_elements(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let params = self.params_for(Modality::Image)?;
        let geometry = params.input_geometry(
            self.pixels.do_resize,
            params.temporal,
            height as usize,
            width as usize,
        )?;
        product(
            &[geometry.target.0, geometry.target.1, 3, params.temporal],
            "GLM image tensor elements",
        )
    }

    /// The budgets for a call of `modality`: this processor's own, or the
    /// error for a call of the modality it was not built for.
    fn params_for(&self, modality: Modality) -> Result<Params, TransformError> {
        if modality == self.modality {
            Ok(self.params)
        } else {
            Err(shape(format!(
                "this GLM processor was built for {} preprocessing, not {modality}",
                self.modality
            )))
        }
    }
}

impl VisionPreProcessor for Glm53FlashProcessor {
    /// Image geometry and normalization depend only on that image's config.
    /// Video frames, in contrast, share a clip budget and temporal grouping.
    fn supports_per_image_preprocessing(&self) -> bool {
        self.modality == Modality::Image
    }

    /// Image BF16 rows and metadata can be split without sharing a clip budget.
    fn supports_bf16_batch_cache(&self) -> bool {
        self.supports_per_image_preprocessing()
    }

    /// Estimate the exact output size using the encoder's resize and padding rules.
    fn image_input_elements(&self, width: u32, height: u32) -> Option<usize> {
        self.image_elements(width, height).ok()
    }

    fn default_mean(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_STD
    }

    /// The patches are the image's own bytes, so an engine that normalizes on
    /// device takes them as they are.
    fn emits_pixel_bytes(&self) -> bool {
        true
    }

    /// Pack independent images into one exactly reserved tensor in request order.
    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let params = self.params_for(Modality::Image)?;
        let norm = self.pixels.norm;
        // Plan the complete batch once instead of reallocating and copying
        // all earlier images whenever the growing output exhausts capacity.
        let expected = images.iter().try_fold(0usize, |total, image| {
            total
                .checked_add(self.image_elements(image.width(), image.height())?)
                .ok_or_else(|| shape("GLM image batch size overflow"))
        })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(expected)
            .map_err(|error| shape(format!("cannot reserve GLM image batch: {error}")))?;
        let mut grids = Vec::with_capacity(images.len() * 3);
        let mut counts = Vec::with_capacity(images.len());
        let mut tokens = Vec::with_capacity(images.len());
        for image in images {
            let patches = encode(
                std::slice::from_ref(*image),
                params.temporal,
                params,
                &self.pixels,
            )?;
            grids.extend([
                patches.grid.0 as i64,
                patches.grid.1 as i64,
                patches.grid.2 as i64,
            ]);
            counts.push(patches.count as i64);
            tokens.push(patches.tokens);
            values.extend(patches.values);
        }
        let features = 3 * params.temporal * params.patch * params.patch;
        let total = values.len() / features;
        let tensor = Array2::from_shape_vec((total, features), values)
            .map_err(|error| shape(format!("failed to build GLM image patches: {error}")))?;
        Ok(PreprocessedEncoderInputs::new(
            EncoderInput::bytes(tensor, norm),
            tokens,
            images
                .iter()
                .copied()
                .map(GenericImageView::dimensions)
                .collect(),
        )
        .with_extra(
            "image_grid_thw",
            ModelSpecificValue::int_2d(grids, images.len(), 3),
        )
        .with_extra("patches_per_image", ModelSpecificValue::int_1d(counts)))
    }

    /// The shared batch cache also reprocesses mixed hit/miss batches. Give
    /// that path wire storage directly so it can retain the newly seen images.
    fn preprocess_wire(
        &self,
        images: &[&DynamicImage],
        dtype: EncoderDtype,
    ) -> Option<Result<PreprocessedEncoderInputs, TransformError>> {
        if self.modality != Modality::Image || dtype != EncoderDtype::BFloat16 {
            return None;
        }
        Some(self.preprocess(images).and_then(|mut output| {
            let dimensions = output.encoder_input.shape().to_vec();
            let bytes = output.encoder_input.into_bytes_as(dtype).map_err(shape)?;
            output.encoder_input =
                EncoderInput::wire(bytes, dtype, dimensions, self.pixels.norm).map_err(shape)?;
            Ok(output)
        }))
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        preprocess_video_frames(
            frames,
            self.params_for(Modality::Video)?,
            &self.pixels,
            clip,
        )
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        preprocess_video_frames(
            frames,
            self.params_for(Modality::Video)?,
            &self.pixels,
            clip,
        )
    }

    /// The tokens of one image, counted on the canvas the encode works on
    /// (the budget search's target, or the source itself under
    /// `do_resize: false`); the preprocessing error for a size it refuses.
    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let params = self.params_for(Modality::Image)?;
        let geometry = params.input_geometry(
            self.pixels.do_resize,
            params.temporal,
            height as usize,
            width as usize,
        )?;
        Ok(
            geometry.target.0 / params.patch * (geometry.target.1 / params.patch)
                / (params.merge * params.merge),
        )
    }

    fn model_name(&self) -> &'static str {
        "glm-5.3-flash"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::test_support::{hex, lcg_bytes};

    /// Cached slices must be exactly the same pixels and metadata as a batch,
    /// including different image sizes, repeated images and custom normalization.
    #[test]
    fn independent_images_and_size_estimates_match_the_batch() {
        use std::collections::HashMap;

        use crate::{EncoderDtype, FieldLayout};

        let layouts = HashMap::from([
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ]);
        let configs = [
            PreProcessorConfig::default(),
            PreProcessorConfig::from_json(
                r#"{"max_image_tokens":32,"patch_expand_factor":2,"do_normalize":false}"#,
            ).unwrap(),
            PreProcessorConfig::from_json(
                r#"{"do_resize":false,"temporal_patch_size":3,"image_mean":[0.1,0.2,0.3],"image_std":[0.2,0.4,0.8]}"#,
            ).unwrap(),
        ];
        let images: Vec<DynamicImage> = [(56, 112), (280, 84), (112, 56)]
            .into_iter()
            .map(|(w, h)| {
                RgbImage::from_fn(w, h, |x, y| {
                    Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
                })
                .into()
            })
            .collect();
        let refs = [&images[2], &images[0], &images[1], &images[2]];
        for config in configs {
            let processor = Glm53FlashProcessor::from_config_for(&config, Modality::Image).unwrap();
            assert!(processor.supports_per_image_preprocessing());
            assert!(processor.supports_bf16_batch_cache());
            let parts: Vec<_> = refs
                .iter()
                .map(|image| {
                    let output = processor.preprocess(&[*image]).unwrap();
                    assert_eq!(
                        processor.image_input_elements(image.width(), image.height()),
                        Some(output.encoder_input.len())
                    );
                    output
                })
                .collect();
            let joined =
                PreprocessedEncoderInputs::concat_refs(&parts.iter().collect::<Vec<_>>(), &layouts)
                    .unwrap();
            let batch = processor.preprocess(&refs).unwrap();
            let wire = processor
                .preprocess_wire(&refs, EncoderDtype::BFloat16)
                .unwrap()
                .unwrap();
            assert!(matches!(
                wire.encoder_input,
                EncoderInput::Wire {
                    dtype: EncoderDtype::BFloat16,
                    ..
                }
            ));
            assert_eq!(
                wire.encoder_input.write_as(EncoderDtype::BFloat16),
                batch.encoder_input.write_as(EncoderDtype::BFloat16)
            );
            assert_eq!(wire.encoder_input.shape(), batch.encoder_input.shape());
            assert_eq!(wire.feature_token_counts, batch.feature_token_counts);
            assert_eq!(wire.item_sizes, batch.item_sizes);
            assert!(processor
                .preprocess_wire(&refs, EncoderDtype::Float32)
                .is_none());
            assert_eq!(joined.encoder_input.shape(), batch.encoder_input.shape());
            assert_eq!(joined.feature_token_counts, batch.feature_token_counts);
            assert_eq!(joined.item_sizes, batch.item_sizes);
            for dtype in [
                EncoderDtype::Uint8,
                EncoderDtype::Float32,
                EncoderDtype::BFloat16,
            ] {
                assert_eq!(
                    joined.encoder_input.write_as(dtype),
                    batch.encoder_input.write_as(dtype)
                );
            }
            for (key, value) in &batch.model_specific {
                assert_eq!(
                    format!("{:?}", joined.model_specific[key]),
                    format!("{value:?}")
                );
            }
            assert_eq!(processor.image_input_elements(0, 112), None);
            let video = Glm53FlashProcessor::from_config_for(&config, Modality::Video).unwrap();
            assert!(!video.supports_per_image_preprocessing());
            assert!(!video.supports_bf16_batch_cache());
            assert_eq!(video.image_input_elements(112, 112), None);
            assert!(video
                .preprocess_wire(&refs, EncoderDtype::BFloat16)
                .is_none());
        }
    }

    fn config(max_tokens: usize) -> PreProcessorConfig {
        PreProcessorConfig::from_json(&format!(
            r#"{{"do_resize":true,"patch_size":14,"merge_size":2,"temporal_patch_size":2,
            "patch_expand_factor":1,"min_image_tokens":16,"max_image_tokens":{max_tokens}}}"#
        ))
        .unwrap()
    }

    fn grid(output: &PreprocessedEncoderInputs, key: &str) -> Vec<i64> {
        match output.model_specific.get(key).unwrap() {
            ModelSpecificValue::IntTensor { data, .. } => data.clone(),
            value => panic!("unexpected grid value: {value:?}"),
        }
    }

    /// A processor is built for images or video; any other modality is
    /// refused when it is built rather than on every call.
    #[test]
    fn construction_refuses_a_modality_it_does_not_preprocess() {
        for modality in [Modality::Audio, Modality::ImageEmbeds] {
            assert!(
                Glm53FlashProcessor::from_config_for(&PreProcessorConfig::default(), modality)
                    .is_err(),
                "{modality}"
            );
        }
    }

    #[test]
    fn construction_reports_an_unusable_expand_factor() {
        // A present but unusable `patch_expand_factor` is the configuration
        // error the parameters report, not a count of zero.
        let mut config = PreProcessorConfig::default();
        config
            .extra
            .insert("patch_expand_factor".to_string(), serde_json::json!("x"));
        let constructed = Glm53FlashProcessor::from_config_for(&config, Modality::Image)
            .err()
            .unwrap()
            .to_string();
        assert_eq!(
            constructed,
            Params::from_config(&config, false)
                .err()
                .unwrap()
                .to_string()
        );
    }

    /// The video processor config this family's checkpoint ships, the keys
    /// the engine's own processor reads.
    fn video_config() -> PreProcessorConfig {
        PreProcessorConfig::from_json(
            r#"{"do_rescale":true,"video_processor_type":"Glm5NextVideoProcessor",
            "patch_expand_factor":1,"merge_size":2,"image_mean":[0.48145466,0.4578275,0.40821073],
            "image_std":[0.26862954,0.26130258,0.27577711],"temporal_patch_size":2,"patch_size":14,
            "min_image_tokens":16,"max_image_tokens":240000,"fps":2}"#,
        )
        .unwrap()
    }

    /// A video config keeps its checkpoint's token budget. An image config
    /// used for video keeps the video default, and explicit video token
    /// budgets take precedence.
    #[test]
    fn video_budget_respects_the_checkpoint_token_budget() {
        let ppt = 2 * 28 * 28;
        let params = Params::from_config(&video_config(), true).unwrap();
        assert_eq!(params.max_pixels, MAX_VIDEO_TOKENS * ppt);
        assert_eq!(params.min_pixels, 16 * ppt);
        assert_eq!(
            params.geometry(32, 2060, 3416).unwrap().target,
            (2072, 3416)
        );

        for (tokens, side) in [(14_400, 3360), (40_000, 5600)] {
            let mut config = video_config();
            config
                .extra
                .insert("max_image_tokens".into(), serde_json::json!(tokens));
            let params = Params::from_config(&config, true).unwrap();
            assert_eq!(params.max_pixels, tokens * ppt);
            assert_eq!(params.geometry(2, 5600, 5600).unwrap().target, (side, side));
        }

        let image_fallback = Params::from_config(&config(MAX_IMAGE_TOKENS), true).unwrap();
        assert_eq!(image_fallback.max_pixels, MAX_VIDEO_TOKENS * ppt);

        let mut explicit = video_config();
        explicit
            .extra
            .insert("max_video_tokens".into(), serde_json::json!(50_000));
        let explicit = Params::from_config(&explicit, true).unwrap();
        assert_eq!(explicit.max_pixels, 50_000 * ppt);
    }

    #[test]
    fn geometry_matches_reference_edge_cases() {
        let params = Params::from_config(&config(MAX_VIDEO_TOKENS), true).unwrap();
        let long_video = params.geometry(512, 2160, 3840).unwrap();
        assert_eq!(long_video.target, (644, 1120));
        assert_eq!(params.geometry(1, 1, 1).unwrap().target, (168, 168));

        let mut expanded = config(MAX_IMAGE_TOKENS);
        expanded
            .extra
            .insert("patch_expand_factor".into(), serde_json::json!(2));
        let expanded = Params::from_config(&expanded, false).unwrap();
        assert_eq!(
            expanded.geometry(expanded.temporal, 1, 1).unwrap().target,
            (224, 224)
        );
    }

    #[test]
    fn image_resize_preserves_content_padding() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(400, 200, Rgb([255, 0, 0])));
        let output =
            Glm53FlashProcessor::from_config_for(&config(MAX_IMAGE_TOKENS), Modality::Image)
                .unwrap()
                .preprocess(&[&image])
                .unwrap();
        assert_eq!(grid(&output, "image_grid_thw"), vec![1, 16, 30]);
        let padding = (-PreProcessorConfig::CLIP_MEAN[0] / PreProcessorConfig::CLIP_STD[0]) as f32;
        assert!(output
            .encoder_input
            .as_f32()
            .iter()
            .any(|value| (*value - padding).abs() < 1e-5));
    }

    #[test]
    fn borrowed_and_dynamic_video_paths_match_layout_and_timing() {
        let rgb = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
            .map(|color| RgbImage::from_pixel(80, 40, Rgb(color)));
        let dynamic = rgb
            .iter()
            .cloned()
            .map(DynamicImage::ImageRgb8)
            .collect::<Vec<_>>();
        let borrowed = rgb
            .iter()
            .map(|frame| RgbFrameRef {
                width: frame.width(),
                height: frame.height(),
                data: frame.as_raw(),
            })
            .collect::<Vec<_>>();
        let mut resize = config(MAX_VIDEO_TOKENS);
        resize.min_pixels = Some(1);
        resize.max_pixels = Some(4 * 28 * 56);
        let resize = Glm53FlashProcessor::from_config_for(&resize, Modality::Video).unwrap();
        let clip = VideoClipOptions::sampled_at(2.0);
        let dynamic = resize.preprocess_video(&dynamic, &clip).unwrap();
        let borrowed = resize.preprocess_video_rgb(&borrowed, &clip).unwrap();
        assert_eq!(dynamic.encoder_input, borrowed.encoder_input);
        assert_eq!(grid(&dynamic, "video_grid_thw"), vec![2, 2, 4]);
        // The entry's video layout slices the clip by `patches_per_image`, the
        // legacy map's one sizes key, so the video output carries that count
        // beside `patches_per_video`: the grid's product, the encoder rows the
        // assembler checks the item sizes against.
        assert_eq!(grid(&dynamic, "patches_per_image"), vec![2 * 2 * 4]);
        assert_eq!(
            grid(&dynamic, "patches_per_image"),
            grid(&dynamic, "patches_per_video")
        );
        assert!(matches!(
            dynamic.model_specific.get("video_second_per_grid"),
            Some(ModelSpecificValue::Tensor { data, .. }) if data == &[1.0]
        ));

        let frames = [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 28, Rgb([10, 20, 30]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 28, Rgb([40, 50, 60]))),
        ];
        let mut no_resize = config(MAX_VIDEO_TOKENS);
        no_resize.do_resize = Some(false);
        no_resize.do_normalize = Some(false);
        let output = Glm53FlashProcessor::from_config_for(&no_resize, Modality::Video)
            .unwrap()
            .preprocess_video(&frames, &clip)
            .unwrap();
        let flat = output.encoder_input.flat_f32().into_owned();
        let patch = &flat[..1176];
        for (offset, value) in [10.0, 40.0, 20.0, 50.0, 30.0, 60.0].into_iter().enumerate() {
            assert!((patch[offset * 196] - value / 255.0).abs() < 1e-6);
        }
    }

    #[test]
    fn rescale_flags_and_invalid_transforms_are_honored() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 28, Rgb([10, 20, 30])));
        let first = |config: &PreProcessorConfig| {
            Glm53FlashProcessor::from_config_for(config, Modality::Image)
                .and_then(|processor| processor.preprocess(&[&image]))
                .map(|output| output.encoder_input.as_f32()[[0, 0]])
        };
        let mut config = config(MAX_IMAGE_TOKENS);
        config.do_resize = Some(false);
        config.do_normalize = Some(false);
        assert!((first(&config).unwrap() - 10.0 / 255.0).abs() < 1e-6);
        config.do_rescale = Some(false);
        assert_eq!(first(&config).unwrap(), 10.0);
        config.do_rescale = Some(true);
        config.rescale_factor = Some(0.5);
        assert_eq!(first(&config).unwrap(), 5.0);

        config.do_normalize = Some(true);
        config.image_std = Some(vec![0.0, 1.0, 1.0]);
        assert!(first(&config).is_err());
        config.image_std = None;
        config.rescale_factor = Some(f64::NAN);
        assert!(first(&config).is_err());
    }

    /// The patches are the image's own bytes, so an engine that normalizes on
    /// device takes them unchanged; a float destination maps each byte the
    /// way transformers' processor does, `(byte - mean / rescale) / (std /
    /// rescale)` in f32.
    #[test]
    fn patches_are_pixel_bytes_normalized_with_the_references_rounding() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 28, Rgb([255, 128, 7])));
        let mut config = config(MAX_IMAGE_TOKENS);
        config.do_resize = Some(false);
        let processor = Glm53FlashProcessor::from_config_for(&config, Modality::Image).unwrap();
        assert!(processor.emits_pixel_bytes());
        let output = processor.preprocess(&[&image]).unwrap();
        let EncoderInput::U8 { patches, norm } = &output.encoder_input else {
            panic!("GLM patches should be pixel bytes");
        };
        let channel_block = TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
        assert_eq!(norm.channel_block, channel_block);
        assert_eq!(patches[[0, 0]], 255);
        assert_eq!(patches[[0, channel_block]], 128);
        assert_eq!(patches[[0, 2 * channel_block]], 7);
        let into_bytes = (1.0 / (1.0 / 255.0_f64)) as f32;
        let reference = |byte: f32, channel: usize| {
            (byte - PreProcessorConfig::CLIP_MEAN[channel] as f32 * into_bytes)
                / (PreProcessorConfig::CLIP_STD[channel] as f32 * into_bytes)
        };
        let floats = output.encoder_input.as_f32();
        assert_eq!(floats[[0, 0]].to_bits(), reference(255.0, 0).to_bits());
        assert_eq!(
            floats[[0, channel_block]].to_bits(),
            reference(128.0, 1).to_bits()
        );
        assert_eq!(
            floats[[0, 2 * channel_block]].to_bits(),
            reference(7.0, 2).to_bits()
        );
        // Every byte of every channel maps to the reference's f32, bit for
        // bit: the table a float destination reads is the whole contract.
        let table = norm.table_f32();
        for (channel, column) in table.iter().enumerate() {
            for (byte, value) in column.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    reference(byte as f32, channel).to_bits(),
                    "channel {channel} byte {byte}"
                );
            }
        }
    }

    #[test]
    fn extreme_aspect_ratio_is_rejected() {
        // Without the guard a 1x250000 strip degenerates to a 1-pixel
        // content on a (factor, factor) canvas instead of erroring.
        let params = Params::from_config(&config(MAX_IMAGE_TOKENS), false).unwrap();
        let err = params.geometry(1, 1, 250_000).unwrap_err();
        assert!(err.to_string().contains("aspect ratio"), "{err}");
        // Just inside the limit still resolves.
        assert!(params.geometry(1, 10, 1990).is_ok());
    }

    #[test]
    fn mistyped_extra_keys_error_instead_of_silently_defaulting() {
        // A wrongly-typed budget key must not silently become the default:
        // the resulting token geometry would desync gateway and engine.
        let mut bad = config(MAX_IMAGE_TOKENS);
        bad.extra
            .insert("max_image_tokens".into(), serde_json::json!("8000"));
        assert!(Params::from_config(&bad, false).is_err());

        // Integral JSON floats are accepted (float-happy toolchains).
        let mut float_expand = config(MAX_IMAGE_TOKENS);
        float_expand
            .extra
            .insert("patch_expand_factor".into(), serde_json::json!(2.0));
        let params = Params::from_config(&float_expand, false).unwrap();
        assert_eq!(params.factor, 14 * 2 * 2, "2.0 must read as expand=2");

        // A non-positive fps fails the video path loudly.
        let frames = [DynamicImage::ImageRgb8(RgbImage::from_pixel(
            56,
            56,
            Rgb([1, 2, 3]),
        ))];
        assert!(
            Glm53FlashProcessor::from_config_for(&config(MAX_VIDEO_TOKENS), Modality::Video)
                .unwrap()
                .preprocess_video(&frames, &VideoClipOptions::sampled_at(0.0))
                .is_err()
        );
    }

    #[test]
    fn video_budget_ignores_image_token_caps() {
        // With no dedicated video config the gateway falls back to the
        // image config; an image-scale max_image_tokens must not cap a
        // clip's volume. pixels_per_token = factor^2 * temporal.
        let ppt = 28 * 28 * 2;
        let capped = config(100);
        let image = Params::from_config(&capped, false).unwrap();
        assert_eq!(image.max_pixels, 100 * ppt);
        let video = Params::from_config(&capped, true).unwrap();
        assert_eq!(video.max_pixels, MAX_VIDEO_TOKENS * ppt);
    }

    #[test]
    fn video_frames_with_mismatched_dimensions_error() {
        let frames = [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(56, 56, Rgb([1, 2, 3]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 56, Rgb([1, 2, 3]))),
        ];
        let err = Glm53FlashProcessor::from_config_for(&config(MAX_VIDEO_TOKENS), Modality::Video)
            .unwrap()
            .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
            .unwrap_err();
        assert!(err.to_string().contains("identical dimensions"), "{err}");
    }

    #[test]
    fn degenerate_budgets_and_unaligned_no_resize_error() {
        // max_pixels smaller than one aligned patch can fit nothing.
        let mut tiny = config(MAX_IMAGE_TOKENS);
        tiny.min_pixels = Some(1);
        tiny.max_pixels = Some(1);
        let params = Params::from_config(&tiny, false).unwrap();
        let err = params.geometry(1, 100, 100).unwrap_err();
        assert!(err.to_string().contains("aligned patch"), "{err}");

        // do_resize=false hands the canvas alignment burden to the caller.
        let mut no_resize = config(MAX_IMAGE_TOKENS);
        no_resize.do_resize = Some(false);
        let unaligned = DynamicImage::ImageRgb8(RgbImage::from_pixel(30, 30, Rgb([1, 2, 3])));
        let no_resize = Glm53FlashProcessor::from_config_for(&no_resize, Modality::Image).unwrap();
        let err = no_resize.preprocess(&[&unaligned]).unwrap_err();
        assert!(err.to_string().contains("not aligned"), "{err}");
        // The size-only count takes the same source canvas as the encode.
        assert_eq!(no_resize.calculate_num_tokens(56, 56).unwrap(), 4);
        assert!(no_resize.calculate_num_tokens(30, 30).is_err());
    }

    #[test]
    fn multi_image_batches_accumulate_grids_and_patches() {
        let images = [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(56, 56, Rgb([10, 20, 30]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(112, 56, Rgb([40, 50, 60]))),
        ];
        let output =
            Glm53FlashProcessor::from_config_for(&config(MAX_IMAGE_TOKENS), Modality::Image)
                .unwrap()
                .preprocess(&images.iter().collect::<Vec<_>>())
                .unwrap();
        let grids = grid(&output, "image_grid_thw");
        assert_eq!(grids.len(), 6, "one [t,h,w] row per image");
        let rows: usize = grids
            .chunks(3)
            .map(|row| (row[0] * row[1] * row[2]) as usize)
            .sum();
        assert_eq!(
            output.encoder_input.shape()[0],
            rows,
            "patch rows must equal the summed grid volumes"
        );
    }

    /// A bicubic resize takes the reference's kernel, torchvision's
    /// antialiased bicubic on the bytes, which its CPU path runs natively on
    /// `u8`. Pillow's bicubic agrees with it at ordinary ratios and parts by
    /// one or two 8-bit levels at the scale extremes, so the pins are two extreme
    /// upscales: the digest of the reference's bytes (torchvision 0.28.0 on
    /// the seeded input) and the positions where Pillow's bytes differ.
    #[test]
    fn prepare_resizes_with_the_reference_kernel_at_extreme_ratios() {
        use sha2::{Digest, Sha256};

        struct Case {
            width: u32,
            height: u32,
            seed: u32,
            // SHA-256 of the reference's 28x28 bytes.
            digest: &'static str,
            // (index, Pillow's byte, the reference's byte) where the two kernels part.
            differing: &'static [(usize, u8, u8)],
        }

        let cases = [
            Case {
                width: 2,
                height: 2,
                seed: 11,
                digest: "58b97a38d107b21c7635038b616e5fd0593204b93b82bdca5ddf413f20c7866d",
                differing: &[
                    (597, 132, 133),
                    (600, 132, 133),
                    (626, 123, 122),
                    (653, 152, 153),
                    (660, 172, 171),
                    (902, 163, 164),
                    (1040, 97, 98),
                    (1292, 85, 84),
                    (1490, 205, 204),
                    (1689, 253, 252),
                    (1692, 253, 252),
                    (1718, 98, 99),
                    (1745, 225, 224),
                    (1752, 3, 4),
                ],
            },
            Case {
                width: 7,
                height: 5,
                seed: 13,
                digest: "aaf8d4df4d527fe8ad579d91011a959e53427463ffcdf8331683256ac2838e7b",
                differing: &[
                    (662, 205, 206),
                    (1157, 155, 154),
                    (1260, 24, 25),
                    (1315, 33, 34),
                    (1702, 89, 90),
                ],
            },
        ];
        for case in cases {
            let Case {
                width,
                height,
                seed,
                digest,
                differing,
            } = case;
            let data = lcg_bytes((width * height * 3) as usize, seed);
            let image = DynamicImage::ImageRgb8(RgbImage::from_raw(width, height, data).unwrap());
            // Content and target both 28x28: the resized bytes are the whole canvas.
            let geometry = Geometry {
                target: (28, 28),
                content: (28, 28),
            };
            let bytes = prepare(&image, geometry, FilterType::CatmullRom, true).unwrap();
            assert_eq!(bytes.len(), 28 * 28 * 3);
            for &(index, pillow, reference) in differing {
                assert_eq!(
                    bytes[index], reference,
                    "{width}x{height} -> 28x28 byte {index} (Pillow gives {pillow})"
                );
            }
            assert_eq!(
                hex(&Sha256::digest(bytes.as_ref())),
                digest,
                "{width}x{height} -> 28x28"
            );
        }
    }
}
