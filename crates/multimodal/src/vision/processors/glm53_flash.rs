//! GLM-5.3-Flash image and video preprocessing.

use std::borrow::Cow;

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array2;

use crate::{
    encoder_inputs::ModelSpecificValue,
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::{
            pil_to_filter, resize_bicubic_pil_rgb, resize_rgb_bytes, rgb_bytes, round_half_to_even,
            TransformError,
        },
    },
};

const PATCH_SIZE: usize = 14;
const MERGE_SIZE: usize = 2;
const TEMPORAL_PATCH_SIZE: usize = 2;
const MIN_IMAGE_TOKENS: usize = 16;
const MAX_IMAGE_TOKENS: usize = 8000;
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

struct Patches {
    values: Vec<f32>,
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

/// Read a `usize` extra key strictly: absent falls back to `default`, but a
/// present-yet-unusable value is a config error. `get_extra` swallows type
/// mismatches into `None`, and a silent default here changes the alignment
/// factor and token geometry — the gateway's placeholder count would then
/// disagree with the serving engine with no trace. Integral JSON floats
/// (`2.0`) are accepted for interop with float-serializing toolchains.
fn extra_usize(
    config: &PreProcessorConfig,
    key: &str,
    default: usize,
) -> Result<usize, TransformError> {
    let Some(value) = config.extra.get(key) else {
        return Ok(default);
    };
    if let Some(n) = value.as_u64() {
        return Ok(n as usize);
    }
    if let Some(f) = value.as_f64() {
        if f >= 0.0 && f.fract() == 0.0 && f <= u32::MAX as f64 {
            return Ok(f as usize);
        }
    }
    Err(shape(format!(
        "GLM preprocessor key {key} must be a non-negative integer, got {value}"
    )))
}

/// `extra_usize`'s float sibling: absent -> default, wrong type -> error.
fn extra_f32(config: &PreProcessorConfig, key: &str, default: f32) -> Result<f32, TransformError> {
    let Some(value) = config.extra.get(key) else {
        return Ok(default);
    };
    value.as_f64().map(|f| f as f32).ok_or_else(|| {
        shape(format!(
            "GLM preprocessor key {key} must be a number, got {value}"
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
        // Video budgets read video-named token keys only: an image-scale
        // `max_image_tokens` must not silently become a clip's volume cap
        // when the gateway falls back to the image config for video.
        let (min_key, max_key, max_fallback) = if video {
            ("min_video_tokens", "max_video_tokens", MAX_VIDEO_TOKENS)
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

fn pixel_lut(
    config: &PreProcessorConfig,
    params: Params,
) -> Result<[[f32; 256]; 3], TransformError> {
    let do_rescale = config.do_rescale.unwrap_or(true);
    let do_normalize = config.do_normalize.unwrap_or(true);
    let rescale = if do_rescale {
        config.rescale_factor.unwrap_or(1.0 / 255.0)
    } else {
        1.0
    };
    if !rescale.is_finite()
        || (do_normalize
            && (params.mean.iter().any(|value| !value.is_finite())
                || params
                    .std
                    .iter()
                    .any(|value| !value.is_finite() || *value == 0.0)))
    {
        return Err(shape("invalid GLM pixel rescale or normalization values"));
    }
    let lut = std::array::from_fn(|channel| {
        let scale = (if do_normalize {
            rescale / params.std[channel]
        } else {
            rescale
        }) as f32;
        let bias = (if do_normalize {
            -params.mean[channel] / params.std[channel]
        } else {
            0.0
        }) as f32;
        std::array::from_fn(|value| value as f32 * scale + bias)
    });
    lut.iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(lut)
        .ok_or_else(|| shape("GLM pixel transform overflow"))
}

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
            resize_bicubic_pil_rgb(
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

fn encode<F: RgbSource>(
    frames: &[F],
    budget_frames: usize,
    params: Params,
    config: &PreProcessorConfig,
) -> Result<Patches, TransformError> {
    let first = frames.first().ok_or(TransformError::EmptyBatch)?;
    let source_size = first.dimensions();
    if frames.iter().any(|frame| frame.dimensions() != source_size) {
        return Err(shape("GLM video frames must have identical dimensions"));
    }
    let do_resize = config.do_resize.unwrap_or(true);
    let geometry = if do_resize {
        params.geometry(
            budget_frames,
            source_size.1 as usize,
            source_size.0 as usize,
        )?
    } else {
        Geometry {
            target: (source_size.1 as usize, source_size.0 as usize),
            content: (source_size.1 as usize, source_size.0 as usize),
        }
    };
    let (height, width) = geometry.target;
    if !height.is_multiple_of(params.factor) || !width.is_multiple_of(params.factor) {
        return Err(shape("GLM canvas is not aligned to the configured factor"));
    }
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
    let lut = pixel_lut(config, params)?;
    let filter = pil_to_filter(config.resampling.or(Some(3)));
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
                        for (channel, channel_lut) in lut.iter().enumerate() {
                            for temporal in 0..params.temporal {
                                let raw = prepared.get(temporal).unwrap_or(last).as_ref();
                                for patch_y in 0..params.patch {
                                    let row = (y0 + merge_h * params.patch + patch_y) * width
                                        + x0
                                        + merge_w * params.patch;
                                    let mut source = row * 3 + channel;
                                    for _ in 0..params.patch {
                                        values.push(channel_lut[raw[source] as usize]);
                                        source += 3;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(values.len(), expected);
    Ok(Patches {
        values,
        grid: (grid_t, grid_h, grid_w),
        count,
        tokens: count / (params.merge * params.merge),
    })
}

fn preprocess_video_frames<F: RgbSource>(
    frames: &[F],
    config: &PreProcessorConfig,
) -> Result<PreprocessedEncoderInputs, TransformError> {
    let original = frames
        .first()
        .ok_or(TransformError::EmptyBatch)?
        .dimensions();
    let params = Params::from_config(config, true)?;
    let patches = encode(frames, frames.len(), params, config)?;
    let fps = extra_f32(config, "fps", 2.0)?;
    if !fps.is_finite() || fps <= 0.0 {
        return Err(shape(format!("GLM video fps must be positive, got {fps}")));
    }
    let (grid_t, grid_h, grid_w) = patches.grid;
    let features = 3 * params.temporal * params.patch * params.patch;
    let tensor = Array2::from_shape_vec((patches.count, features), patches.values)
        .map_err(|error| shape(format!("failed to build GLM video patches: {error}")))?;
    Ok(
        PreprocessedEncoderInputs::new(tensor, vec![patches.tokens], vec![original])
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
                    data: vec![params.temporal as f32 / fps],
                    shape: vec![1],
                },
            ),
    )
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

#[derive(Debug, Clone, Copy, Default)]
pub struct Glm53FlashProcessor;

impl Glm53FlashProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl VisionPreProcessor for Glm53FlashProcessor {
    fn default_mean(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let params = Params::from_config(config, false)?;
        let mut values = Vec::new();
        let mut grids = Vec::with_capacity(images.len() * 3);
        let mut counts = Vec::with_capacity(images.len());
        let mut tokens = Vec::with_capacity(images.len());
        for image in images {
            let patches = encode(std::slice::from_ref(image), params.temporal, params, config)?;
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
            tensor,
            tokens,
            images.iter().map(GenericImageView::dimensions).collect(),
        )
        .with_extra(
            "image_grid_thw",
            ModelSpecificValue::int_2d(grids, images.len(), 3),
        )
        .with_extra("patches_per_image", ModelSpecificValue::int_1d(counts)))
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        preprocess_video_frames(frames, config)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        preprocess_video_frames(frames, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let Ok(params) = Params::from_config(config, false) else {
            return 0;
        };
        let Ok(geometry) = params.geometry(params.temporal, height as usize, width as usize) else {
            return 0;
        };
        geometry.target.0 / params.patch * (geometry.target.1 / params.patch)
            / (params.merge * params.merge)
    }

    fn model_name(&self) -> &'static str {
        "glm-5.3-flash"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

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
        let output = Glm53FlashProcessor::new()
            .preprocess(&[image], &config(MAX_IMAGE_TOKENS))
            .unwrap();
        assert_eq!(grid(&output, "image_grid_thw"), vec![1, 16, 30]);
        let padding = (-PreProcessorConfig::CLIP_MEAN[0] / PreProcessorConfig::CLIP_STD[0]) as f32;
        assert!(output
            .encoder_input
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
        let dynamic = Glm53FlashProcessor::new()
            .preprocess_video(&dynamic, &resize)
            .unwrap();
        let borrowed = Glm53FlashProcessor::new()
            .preprocess_video_rgb(&borrowed, &resize)
            .unwrap();
        assert_eq!(dynamic.encoder_input, borrowed.encoder_input);
        assert_eq!(grid(&dynamic, "video_grid_thw"), vec![2, 2, 4]);
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
        let output = Glm53FlashProcessor::new()
            .preprocess_video(&frames, &no_resize)
            .unwrap();
        let patch = &output.encoder_input.as_slice().unwrap()[..1176];
        for (offset, value) in [10.0, 40.0, 20.0, 50.0, 30.0, 60.0].into_iter().enumerate() {
            assert!((patch[offset * 196] - value / 255.0).abs() < 1e-6);
        }
    }

    #[test]
    fn rescale_flags_and_invalid_transforms_are_honored() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 28, Rgb([10, 20, 30])));
        let first = |config: &PreProcessorConfig| {
            Glm53FlashProcessor::new()
                .preprocess(std::slice::from_ref(&image), config)
                .map(|output| output.encoder_input[[0, 0]])
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

        // Non-numeric fps fails the video path loudly.
        let mut bad_fps = config(MAX_VIDEO_TOKENS);
        bad_fps.extra.insert("fps".into(), serde_json::json!("2"));
        let frames = [DynamicImage::ImageRgb8(RgbImage::from_pixel(
            56,
            56,
            Rgb([1, 2, 3]),
        ))];
        assert!(Glm53FlashProcessor::new()
            .preprocess_video(&frames, &bad_fps)
            .is_err());
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
        let err = Glm53FlashProcessor::new()
            .preprocess_video(&frames, &config(MAX_VIDEO_TOKENS))
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
        let err = Glm53FlashProcessor::new()
            .preprocess(&[unaligned], &no_resize)
            .unwrap_err();
        assert!(err.to_string().contains("not aligned"), "{err}");
    }

    #[test]
    fn multi_image_batches_accumulate_grids_and_patches() {
        let images = [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(56, 56, Rgb([10, 20, 30]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(112, 56, Rgb([40, 50, 60]))),
        ];
        let output = Glm53FlashProcessor::new()
            .preprocess(&images, &config(MAX_IMAGE_TOKENS))
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
}
