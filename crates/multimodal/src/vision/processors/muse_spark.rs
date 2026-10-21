//! Muse Spark image and video preprocessing.
//!
//! The family's reference image and video processors share one pipeline:
//! the image (or every frame of a clip) is resized to the patch grid whose
//! aspect ratio is closest to the input's under a merged-token cap
//! (`smart_resize`), with the torchvision `uint8` antialiased Lanczos
//! resample; the bytes are rescaled and normalized in one fused step; and the
//! pixels are laid out as flat patch rows of `temporal x channel x patch x
//! patch` values, the image repeated on every temporal slot, a clip's frames
//! paired up with the last frame repeated to fill the final pair. The engine
//! reads the rows with one `[t, h, w]` grid per item and merges `merge_size²`
//! neighbouring patches into one token, so an item costs `t * h * w /
//! merge_size²` tokens.
//!
//! Every parameter comes from the checkpoint's processor config (its
//! `image_processor` section for images, `video_processor` for clips):
//! `patch_size`, `merge_size`, `temporal_patch_size`, `max_image_tokens`
//! (images) or `max_video_frame_tokens` (each frame of a clip), the mean, std
//! and rescale factor. Absent values take the reference classes' defaults.
//!
//! The patch rows are produced as the image's own bytes with their
//! normalization ([`EncoderInput::U8`]), so the destination's dtype is written
//! once at serialization and a cached item serves every destination.

use image::DynamicImage;
use ndarray::Array2;

use super::qwen_vl_base::pixel_norm;
use crate::{
    encoder_inputs::{EncoderInput, ModelSpecificValue, PixelNorm, PreprocessedEncoderInputs},
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::VisionPreProcessor,
        transforms::{resize_lanczos_torch_aa_rgb_bytes, rgb_bytes, TransformError},
    },
};

/// Spatial patch size of the reference image and video processors.
pub const DEFAULT_PATCH_SIZE: usize = 14;
/// Patches merged into one token along each spatial axis.
pub const DEFAULT_MERGE_SIZE: usize = 2;
/// Frames per temporal patch.
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;
/// The reference image processor's merged-token cap per image.
pub const DEFAULT_MAX_IMAGE_TOKENS: usize = 4096;
/// The reference video processor class's merged-token cap per frame; the
/// released checkpoints raise it in their processor config.
pub const DEFAULT_MAX_VIDEO_FRAME_TOKENS: usize = 144;
/// Normalization mean of the family.
pub const MUSE_SPARK_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
/// Normalization std of the family.
pub const MUSE_SPARK_STD: [f64; 3] = [0.5, 0.5, 0.5];

const CHANNELS: usize = 3;

/// The processor's view of one preprocessor config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MuseSparkVisionSettings {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    /// Merged-token cap: per image for an image config, per frame for a
    /// video config.
    pub max_tokens: usize,
}

impl MuseSparkVisionSettings {
    /// The settings of an image config: the cap is `max_image_tokens`.
    pub fn for_images(config: &PreProcessorConfig) -> Self {
        Self::read(
            config,
            config
                .get_extra::<usize>("max_image_tokens")
                .unwrap_or(DEFAULT_MAX_IMAGE_TOKENS),
        )
    }

    /// The settings of a video config: the cap is `max_video_frame_tokens`,
    /// applied to every frame.
    pub fn for_video(config: &PreProcessorConfig) -> Self {
        Self::read(
            config,
            config
                .get_extra::<usize>("max_video_frame_tokens")
                .unwrap_or(DEFAULT_MAX_VIDEO_FRAME_TOKENS),
        )
    }

    fn read(config: &PreProcessorConfig, max_tokens: usize) -> Self {
        Self {
            patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE).max(1),
            merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE).max(1),
            temporal_patch_size: config
                .temporal_patch_size
                .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE)
                .max(1),
            max_tokens: max_tokens.max(1),
        }
    }

    /// Pixels per merged token along each axis.
    fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Values in one patch row: `temporal x channel x patch x patch`.
    fn patch_features(&self) -> usize {
        self.temporal_patch_size * CHANNELS * self.patch_size * self.patch_size
    }

    /// The resize target of a `height` x `width` input and the patch grid
    /// behind it.
    fn plan(&self, height: usize, width: usize) -> Result<ResizePlan, TransformError> {
        if height == 0 || width == 0 {
            return Err(TransformError::ShapeError(format!(
                "Muse Spark cannot resize an empty {width}x{height} image"
            )));
        }
        let (grid_rows, grid_cols) =
            smart_resize_grid(height, width, self.factor(), self.max_tokens);
        Ok(ResizePlan {
            height: grid_rows * self.factor(),
            width: grid_cols * self.factor(),
            grid_h: grid_rows * self.merge_size,
            grid_w: grid_cols * self.merge_size,
            tokens: grid_rows * grid_cols,
        })
    }
}

/// One item's resize target and patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizePlan {
    /// Resized height in pixels.
    pub height: usize,
    /// Resized width in pixels.
    pub width: usize,
    /// Patch rows (`h` of the grid).
    pub grid_h: usize,
    /// Patch columns (`w` of the grid).
    pub grid_w: usize,
    /// Merged tokens per temporal group.
    pub tokens: usize,
}

/// The reference `smart_resize`, in merged-token units: the grid of
/// `factor`-pixel cells (rows, columns) whose aspect ratio is closest to the
/// input's among the floor/ceil roundings of the ideal grid, under
/// `max_tokens` cells. The ideal grid is the input divided by `factor`,
/// shrunk by the square root of the excess when it is over the cap.
///
/// The reference builds its candidates as `list(set(product(...)))` and takes
/// the first one of minimal distortion; the set's iteration order, decided by
/// CPython's hash table, is reproduced here because near-square inputs have
/// two grids of identical distortion. `max_tokens / ratio` is raised to `0.5`
/// through the C library's `pow`, as the interpreter does.
pub fn smart_resize_grid(
    height: usize,
    width: usize,
    factor: usize,
    max_tokens: usize,
) -> (usize, usize) {
    let factor = factor.max(1) as f64;
    let mut ideal_h = height as f64 / factor;
    let mut ideal_w = width as f64 / factor;
    let ratio = if ideal_h > 0.0 {
        ideal_w / ideal_h
    } else {
        1.0
    };
    if ideal_h * ideal_w > max_tokens as f64 {
        ideal_h = (max_tokens as f64 / ratio).powf(0.5);
        ideal_w = ideal_h * ratio;
    }
    let product = [
        (ideal_h.floor() as usize, ideal_w.floor() as usize),
        (ideal_h.floor() as usize, ideal_w.ceil() as usize),
        (ideal_h.ceil() as usize, ideal_w.floor() as usize),
        (ideal_h.ceil() as usize, ideal_w.ceil() as usize),
    ];
    let aspect = height as f64 / width as f64;
    let mut best: Option<(f64, usize, usize)> = None;
    for (rows, cols) in cpython_set_order(&product) {
        if rows < 1 || cols < 1 || rows.saturating_mul(cols) > max_tokens {
            continue;
        }
        let distortion = (rows as f64 / cols as f64 - aspect).abs();
        if best.is_none_or(|(current, _, _)| distortion < current) {
            best = Some((distortion, rows, cols));
        }
    }
    match best {
        Some((_, rows, cols)) => (rows, cols),
        // The reference's fallback: the ideal grid rounded half to even, at
        // least one cell each way.
        None => (
            (python_round(ideal_h) as usize).max(1),
            (python_round(ideal_w) as usize).max(1),
        ),
    }
}

/// Python's `round(x)`: half to even.
fn python_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// CPython's hash of a 2-tuple of small non-negative ints (3.8 and later:
/// the xxHash-style combine in `tuplehash`).
fn cpython_pair_hash(a: usize, b: usize) -> u64 {
    const XXPRIME_1: u64 = 11_400_714_785_074_694_791;
    const XXPRIME_2: u64 = 14_029_467_366_897_019_727;
    const XXPRIME_5: u64 = 2_870_177_450_012_600_261;
    let mut acc = XXPRIME_5;
    for lane in [a as u64, b as u64] {
        acc = acc
            .wrapping_add(lane.wrapping_mul(XXPRIME_2))
            .rotate_left(31)
            .wrapping_mul(XXPRIME_1);
    }
    acc = acc.wrapping_add(2 ^ (XXPRIME_5 ^ 3_527_539));
    if acc == u64::MAX {
        1_546_275_796
    } else {
        acc
    }
}

/// Iteration order of `set(items)` in CPython for at most four distinct
/// pairs: an 8-slot table, home slot `hash & 7`, on collision
/// `i = i * 5 + 1 + (perturb >>= 5)`, read back in slot order.
fn cpython_set_order(items: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut table: [Option<(usize, usize)>; 8] = [None; 8];
    for &item in items {
        if table.contains(&Some(item)) {
            continue;
        }
        let hash = cpython_pair_hash(item.0, item.1);
        let mut slot = (hash & 7) as usize;
        let mut perturb = hash;
        while table[slot].is_some() {
            perturb >>= 5;
            slot = ((slot as u64)
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb)
                & 7) as usize;
        }
        table[slot] = Some(item);
    }
    table.iter().flatten().copied().collect()
}

/// A decoded frame as interleaved RGB bytes, the way the reference loader
/// hands the image to the processor (Pillow's `convert("RGB")`: an alpha
/// channel is dropped and every pixel keeps its stored colour, 16-bit
/// greyscale is clipped to 255).
fn pil_rgb_bytes(image: &DynamicImage) -> (usize, usize, std::borrow::Cow<'_, [u8]>) {
    match image {
        DynamicImage::ImageLuma16(wide) => {
            let (width, height) = wide.dimensions();
            let mut out = Vec::with_capacity(width as usize * height as usize * CHANNELS);
            for &sample in wide.as_raw() {
                let value = sample.min(255) as u8;
                out.extend_from_slice(&[value, value, value]);
            }
            (
                width as usize,
                height as usize,
                std::borrow::Cow::Owned(out),
            )
        }
        other => rgb_bytes(other),
    }
}

/// One frame of `plan.width` x `plan.height` interleaved RGB bytes.
struct Frame<'a> {
    data: std::borrow::Cow<'a, [u8]>,
}

/// `data` (`width` x `height` interleaved RGB) at the plan's size: the
/// reference's Lanczos resample, or the bytes themselves when the size
/// already matches (the reference's resize is the identity then).
fn resized_frame<'a>(
    data: std::borrow::Cow<'a, [u8]>,
    width: usize,
    height: usize,
    plan: ResizePlan,
    do_resize: bool,
) -> Result<Frame<'a>, TransformError> {
    if width == plan.width && height == plan.height {
        return Ok(Frame { data });
    }
    if !do_resize {
        return Err(TransformError::ShapeError(format!(
            "Muse Spark needs {}x{} pixels for its patch grid, got {width}x{height} with resizing off",
            plan.width, plan.height
        )));
    }
    let resized = resize_lanczos_torch_aa_rgb_bytes(
        &data,
        width as u32,
        height as u32,
        plan.width as u32,
        plan.height as u32,
    )?;
    Ok(Frame {
        data: std::borrow::Cow::Owned(resized),
    })
}

/// The three channel planes of `width` x `height` interleaved RGB bytes.
fn planes(data: &[u8], width: usize, height: usize) -> [Vec<u8>; 3] {
    let pixels = width * height;
    let mut planes = [vec![0u8; pixels], vec![0u8; pixels], vec![0u8; pixels]];
    let [red, green, blue] = &mut planes;
    for (((pixel, r), g), b) in data
        .as_chunks::<CHANNELS>()
        .0
        .iter()
        .zip(red.iter_mut())
        .zip(green.iter_mut())
        .zip(blue.iter_mut())
    {
        *r = pixel[0];
        *g = pixel[1];
        *b = pixel[2];
    }
    planes
}

/// Lay `frames` (one temporal group, up to `temporal_patch_size` frames of
/// `plan.width` x `plan.height` interleaved RGB; the last repeated for the
/// slots the group did not fill) out as patch rows into `out`, which must
/// hold `grid_h * grid_w * patch_features` bytes: for every patch its
/// temporal slots, each slot its channels, each channel the patch's rows.
/// Each frame is split into channel planes once, so every patch row is a
/// handful of contiguous copies; a frame that fills several slots is copied
/// into the first and the row duplicated from there.
fn patchify_group(
    frames: &[&[u8]],
    plan: ResizePlan,
    settings: MuseSparkVisionSettings,
    out: &mut [u8],
) {
    let patch = settings.patch_size;
    let channel_block = patch * patch;
    let slot_block = CHANNELS * channel_block;
    let features = settings.patch_features();
    debug_assert_eq!(out.len(), plan.grid_h * plan.grid_w * features);
    debug_assert!(!frames.is_empty());
    let frame_planes: Vec<[Vec<u8>; 3]> = frames
        .iter()
        .map(|frame| planes(frame, plan.width, plan.height))
        .collect();
    let width = plan.width;
    for (row_index, row) in out.chunks_exact_mut(features).enumerate() {
        let patch_row = row_index / plan.grid_w;
        let patch_col = row_index % plan.grid_w;
        let top = patch_row * patch;
        let left = patch_col * patch;
        for temporal in 0..settings.temporal_patch_size {
            let slot = &mut row[temporal * slot_block..(temporal + 1) * slot_block];
            match frame_planes.get(temporal) {
                Some(planes) => {
                    for (channel, plane) in planes.iter().enumerate() {
                        let block =
                            &mut slot[channel * channel_block..(channel + 1) * channel_block];
                        for (y, line) in block.chunks_exact_mut(patch).enumerate() {
                            let start = (top + y) * width + left;
                            line.copy_from_slice(&plane[start..start + patch]);
                        }
                    }
                }
                None => {
                    // The last frame fills the remaining slots: the slot just
                    // written is the one to repeat.
                    let previous = temporal - 1;
                    let (before, rest) = row.split_at_mut(temporal * slot_block);
                    rest[..slot_block].copy_from_slice(
                        &before[previous * slot_block..(previous + 1) * slot_block],
                    );
                }
            }
        }
    }
}

/// Muse Spark image and video processor.
#[derive(Debug, Clone, Copy, Default)]
pub struct MuseSparkVisionProcessor;

impl MuseSparkVisionProcessor {
    pub fn new() -> Self {
        Self
    }

    fn norm(config: &PreProcessorConfig, settings: &MuseSparkVisionSettings) -> PixelNorm {
        // Channel `c` owns the `c`-th block of `patch x patch` values of every
        // temporal slot: the blocks cycle through the channels along the row.
        pixel_norm(
            config,
            MUSE_SPARK_MEAN,
            MUSE_SPARK_STD,
            settings.patch_size * settings.patch_size,
        )
    }

    /// The resize plan of a `width` x `height` image under `config`.
    pub fn image_plan(
        width: u32,
        height: u32,
        config: &PreProcessorConfig,
    ) -> Result<ResizePlan, TransformError> {
        MuseSparkVisionSettings::for_images(config).plan(height as usize, width as usize)
    }

    /// The resize plan of a clip's `width` x `height` frames under `config`.
    pub fn video_plan(
        width: u32,
        height: u32,
        config: &PreProcessorConfig,
    ) -> Result<ResizePlan, TransformError> {
        MuseSparkVisionSettings::for_video(config).plan(height as usize, width as usize)
    }

    fn preprocess_frames<'a, I>(
        frames: I,
        count: usize,
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError>
    where
        I: Iterator<Item = (usize, usize, std::borrow::Cow<'a, [u8]>)>,
    {
        if count == 0 {
            return Err(TransformError::EmptyBatch);
        }
        let settings = MuseSparkVisionSettings::for_video(config);
        let do_resize = config.do_resize.unwrap_or(true);
        let features = settings.patch_features();
        let temporal = settings.temporal_patch_size;
        let groups = count.div_ceil(temporal);
        let mut plan: Option<ResizePlan> = None;
        let mut original = (0u32, 0u32);
        let mut rows_per_group = 0;
        let mut out: Vec<u8> = Vec::new();
        let mut group: Vec<Vec<u8>> = Vec::with_capacity(temporal);
        let mut written = 0;
        for (index, (width, height, data)) in frames.enumerate() {
            let plan = match plan {
                Some(plan) => plan,
                None => {
                    let first = settings.plan(height, width)?;
                    original = (width as u32, height as u32);
                    rows_per_group = first.grid_h * first.grid_w;
                    out = vec![0u8; groups * rows_per_group * features];
                    plan = Some(first);
                    first
                }
            };
            if (width as u32, height as u32) != original {
                return Err(TransformError::InconsistentShapes);
            }
            let frame = resized_frame(data, width, height, plan, do_resize)?;
            group.push(frame.data.into_owned());
            let last_of_group = group.len() == temporal || index + 1 == count;
            if last_of_group {
                let refs: Vec<&[u8]> = group.iter().map(Vec::as_slice).collect();
                let slice =
                    &mut out[written * rows_per_group * features..][..rows_per_group * features];
                patchify_group(&refs, plan, settings, slice);
                written += 1;
                group.clear();
            }
        }
        let plan = plan.ok_or(TransformError::EmptyBatch)?;
        let num_rows = groups * rows_per_group;
        let encoder_input = Array2::from_shape_vec((num_rows, features), out).map_err(|error| {
            TransformError::ShapeError(format!(
                "Muse Spark video encoder_input [{num_rows}, {features}]: {error}"
            ))
        })?;
        Ok(PreprocessedEncoderInputs::new(
            EncoderInput::bytes(encoder_input, Self::norm(config, &settings)),
            vec![groups * plan.tokens],
            vec![original],
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(
                vec![groups as i64, plan.grid_h as i64, plan.grid_w as i64],
                1,
                3,
            ),
        )
        .with_extra(
            "patches_per_video",
            ModelSpecificValue::int_1d(vec![num_rows as i64]),
        ))
    }
}

impl VisionPreProcessor for MuseSparkVisionProcessor {
    fn supports_per_image_preprocessing(&self) -> bool {
        // Every image is planned, resized and laid out on its own.
        true
    }

    fn emits_pixel_bytes(&self) -> bool {
        true
    }

    /// The reference loader hands the processor the image as stored, after
    /// Pillow's `convert("RGB")`; the conversion happens here.
    fn keeps_alpha(&self) -> bool {
        true
    }

    fn default_mean(&self) -> [f64; 3] {
        MUSE_SPARK_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        MUSE_SPARK_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let settings = MuseSparkVisionSettings::for_images(config);
        let do_resize = config.do_resize.unwrap_or(true);
        let features = settings.patch_features();
        let mut plans = Vec::with_capacity(images.len());
        let mut total_rows = 0;
        for image in images {
            let (width, height) = (image.width() as usize, image.height() as usize);
            let plan = settings.plan(height, width)?;
            total_rows += plan.grid_h * plan.grid_w;
            plans.push(plan);
        }
        let mut out = vec![0u8; total_rows * features];
        let mut tokens = Vec::with_capacity(images.len());
        let mut sizes = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len() * 3);
        let mut rows = Vec::with_capacity(images.len());
        let mut written = 0;
        for (image, plan) in images.iter().zip(&plans) {
            let (width, height, data) = pil_rgb_bytes(image);
            let frame = resized_frame(data, width, height, *plan, do_resize)?;
            let item_rows = plan.grid_h * plan.grid_w;
            patchify_group(
                &[&frame.data],
                *plan,
                settings,
                &mut out[written * features..][..item_rows * features],
            );
            written += item_rows;
            tokens.push(plan.tokens);
            sizes.push((width as u32, height as u32));
            grids.extend([1, plan.grid_h as i64, plan.grid_w as i64]);
            rows.push(item_rows as i64);
        }
        let encoder_input =
            Array2::from_shape_vec((total_rows, features), out).map_err(|error| {
                TransformError::ShapeError(format!(
                    "Muse Spark image encoder_input [{total_rows}, {features}]: {error}"
                ))
            })?;
        Ok(PreprocessedEncoderInputs::new(
            EncoderInput::bytes(encoder_input, Self::norm(config, &settings)),
            tokens,
            sizes,
        )
        .with_extra(
            "image_grid_thw",
            ModelSpecificValue::int_2d(grids, images.len(), 3),
        )
        .with_extra("patches_per_image", ModelSpecificValue::int_1d(rows)))
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Self::preprocess_frames(frames.iter().map(pil_rgb_bytes), frames.len(), config)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        for frame in frames {
            if frame.data.len() != frame.width as usize * frame.height as usize * CHANNELS {
                return Err(TransformError::ShapeError(format!(
                    "Muse Spark RGB frame holds {} bytes for {}x{}",
                    frame.data.len(),
                    frame.width,
                    frame.height
                )));
            }
        }
        Self::preprocess_frames(
            frames.iter().map(|frame| {
                (
                    frame.width as usize,
                    frame.height as usize,
                    std::borrow::Cow::Borrowed(frame.data),
                )
            }),
            frames.len(),
            config,
        )
    }

    fn calculate_num_tokens(
        &self,
        width: u32,
        height: u32,
        config: &PreProcessorConfig,
    ) -> Result<usize, TransformError> {
        Ok(Self::image_plan(width, height, config)?.tokens)
    }

    fn model_name(&self) -> &'static str {
        "muse_spark"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::encoder_inputs::EncoderDtype;

    /// The released checkpoints' `processor_config.json` sections.
    const IMAGE_CONFIG: &str = r#"{
        "do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "do_resize": true,
        "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "MuseGlimmerImageProcessor",
        "image_std": [0.5, 0.5, 0.5], "max_image_tokens": 4096, "merge_size": 2,
        "patch_size": 14, "resample": 1, "rescale_factor": 0.00392156862745098,
        "temporal_patch_size": 2
    }"#;
    const VIDEO_CONFIG: &str = r#"{
        "do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "do_resize": true,
        "do_sample_frames": true, "fps": 2.0, "frame_timestamp_format": "Time: {timestamp:.2f}s",
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "max_video_frame_tokens": 256,
        "merge_size": 2, "num_frames": 600, "patch_size": 14, "resample": 1,
        "rescale_factor": 0.00392156862745098, "return_metadata": true, "temporal_patch_size": 2,
        "video_processor_type": "MuseGlimmerVideoProcessor"
    }"#;

    fn image_config() -> PreProcessorConfig {
        PreProcessorConfig::from_json(IMAGE_CONFIG).unwrap()
    }

    fn video_config() -> PreProcessorConfig {
        PreProcessorConfig::from_json(VIDEO_CONFIG).unwrap()
    }

    fn solid(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn settings_come_from_the_processor_config() {
        let images = MuseSparkVisionSettings::for_images(&image_config());
        assert_eq!(
            images,
            MuseSparkVisionSettings {
                patch_size: 14,
                merge_size: 2,
                temporal_patch_size: 2,
                max_tokens: 4096,
            }
        );
        let video = MuseSparkVisionSettings::for_video(&video_config());
        assert_eq!(video.max_tokens, 256);
        // An empty config takes the reference classes' defaults.
        let defaults = MuseSparkVisionSettings::for_video(&PreProcessorConfig::default());
        assert_eq!(defaults.max_tokens, DEFAULT_MAX_VIDEO_FRAME_TOKENS);
        assert_eq!(
            MuseSparkVisionSettings::for_images(&PreProcessorConfig::default()).max_tokens,
            DEFAULT_MAX_IMAGE_TOKENS
        );
    }

    /// Grids from the reference `smart_resize` (patch 28, cap 4096): inputs
    /// under the cap round each axis independently; a near-square input has
    /// two grids of equal distortion and the set order decides.
    #[test]
    fn smart_resize_matches_the_reference_grids() {
        for (height, width, want) in [
            (480, 640, (17, 23)),
            (426, 640, (15, 23)),
            (64, 64, (3, 3)),
            (1, 1, (1, 1)),
            (720, 1280, (26, 46)),
            (100, 60, (3, 2)),
            (61, 99, (2, 3)),
            (300, 200, (11, 7)),
            (130, 137, (4, 4)),
            (469, 463, (17, 17)),
            (655, 648, (23, 23)),
            (688, 698, (25, 25)),
            // Over the cap: the ideal grid shrinks to the budget.
            (4000, 4000, (64, 64)),
            (2000, 8000, (32, 128)),
        ] {
            assert_eq!(
                smart_resize_grid(height, width, 28, 4096),
                want,
                "{height}x{width}"
            );
        }
        // The per-frame cap of the released video config.
        for (height, width, want) in [
            (480, 640, (14, 18)),
            (426, 640, (13, 19)),
            (720, 1280, (12, 21)),
            (469, 463, (16, 16)),
            (2000, 8000, (8, 32)),
        ] {
            assert_eq!(
                smart_resize_grid(height, width, 28, 256),
                want,
                "{height}x{width} at 256"
            );
        }
    }

    #[test]
    fn cpython_pair_hash_matches_the_interpreter() {
        // `hash((a, b)) & (2**64 - 1)` from CPython 3.12.
        assert_eq!(cpython_pair_hash(1, 2), 14_896_688_948_223_909_699);
        assert_eq!(cpython_pair_hash(10, 7), 3_571_329_324_243_458_413);
        assert_eq!(cpython_pair_hash(11, 7), 9_023_120_593_223_478_337);
        assert_eq!(cpython_pair_hash(4, 4), 3_883_364_387_212_965_600);
        assert_eq!(cpython_pair_hash(5, 5), 2_283_807_709_307_846_243);
        // `list(set([(10, 7), (10, 8), (11, 7), (11, 8)]))` in CPython 3.12:
        // (11, 7) comes before (10, 7), which is why 300x200 resolves to an
        // 11-row grid although both grids distort the aspect equally.
        assert_eq!(
            cpython_set_order(&[(10, 7), (10, 8), (11, 7), (11, 8)]),
            vec![(11, 7), (11, 8), (10, 7), (10, 8)]
        );
    }

    #[test]
    fn image_rows_are_temporal_channel_row_column_and_normalized_in_f32() {
        let image = solid(28, 28, Rgb([255, 128, 0]));
        let output = MuseSparkVisionProcessor::new()
            .preprocess(&[image], &image_config())
            .unwrap();
        assert_eq!(output.encoder_input.shape(), &[4, 1176]);
        assert_eq!(output.feature_token_counts, vec![1]);
        assert_eq!(output.item_sizes, vec![(28, 28)]);
        let values = output.encoder_input.as_f32().into_owned();
        let row = values.index_axis(ndarray::Axis(0), 0);
        let norm = |byte: f32| (byte - 127.5) / 127.5;
        assert_eq!(row[0], norm(255.0));
        assert_eq!(row[196], norm(128.0));
        assert_eq!(row[392], norm(0.0));
        // The second temporal slot repeats the image.
        assert_eq!(row[588], row[0]);
        assert_eq!(row[784], row[196]);
        assert!(matches!(
            output.model_specific.get("image_grid_thw"),
            Some(ModelSpecificValue::IntTensor { data, shape }) if data == &[1, 2, 2] && shape == &[1, 3]
        ));
        assert!(matches!(
            output.model_specific.get("patches_per_image"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &[4]
        ));
        // The bfloat16 wire rounds the same f32 values to nearest even.
        let wire = output.encoder_input.write_as(EncoderDtype::BFloat16);
        assert_eq!(wire.len(), 4 * 1176 * 2);
    }

    #[test]
    fn patches_follow_the_image_in_reading_order() {
        // 56x28: two patch columns, the left half red and the right half blue.
        let mut image = RgbImage::from_pixel(56, 28, Rgb([200, 0, 0]));
        for y in 0..28 {
            for x in 28..56 {
                image.put_pixel(x, y, Rgb([0, 0, 200]));
            }
        }
        let output = MuseSparkVisionProcessor::new()
            .preprocess(&[DynamicImage::ImageRgb8(image)], &image_config())
            .unwrap();
        assert_eq!(output.encoder_input.shape(), &[8, 1176]);
        let values = output.encoder_input.as_f32().into_owned();
        // Rows are row-major over the patch grid: columns 0, 1 red then
        // columns 2, 3 blue, for each of the two patch rows.
        for (index, red) in [true, true, false, false, true, true, false, false]
            .into_iter()
            .enumerate()
        {
            let row = values.index_axis(ndarray::Axis(0), index);
            let (r, b) = (row[0], row[392]);
            if red {
                assert!(r > 0.0 && b < 0.0, "row {index}");
            } else {
                assert!(r < 0.0 && b > 0.0, "row {index}");
            }
        }
    }

    #[test]
    fn alpha_is_dropped_and_sixteen_bit_grey_is_clipped_like_the_reference_loader() {
        let rgba = image::RgbaImage::from_pixel(28, 28, image::Rgba([10, 20, 30, 0]));
        let output = MuseSparkVisionProcessor::new()
            .preprocess(&[DynamicImage::ImageRgba8(rgba)], &image_config())
            .unwrap();
        let values = output.encoder_input.as_f32().into_owned();
        let row = values.index_axis(ndarray::Axis(0), 0);
        assert_eq!(row[0], (10.0 - 127.5) / 127.5);
        assert_eq!(row[196], (20.0 - 127.5) / 127.5);

        let grey = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_pixel(
            28,
            28,
            image::Luma([300u16]),
        );
        let output = MuseSparkVisionProcessor::new()
            .preprocess(&[DynamicImage::ImageLuma16(grey)], &image_config())
            .unwrap();
        let values = output.encoder_input.as_f32().into_owned();
        assert_eq!(values[[0, 0]], 1.0);
        let grey = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_pixel(
            28,
            28,
            image::Luma([100u16]),
        );
        let output = MuseSparkVisionProcessor::new()
            .preprocess(&[DynamicImage::ImageLuma16(grey)], &image_config())
            .unwrap();
        let values = output.encoder_input.as_f32().into_owned();
        assert_eq!(values[[0, 0]], (100.0 - 127.5) / 127.5);
    }

    #[test]
    fn batched_images_equal_their_single_preprocesses() {
        let processor = MuseSparkVisionProcessor::new();
        let config = image_config();
        let a = solid(100, 60, Rgb([1, 2, 3]));
        let b = solid(61, 99, Rgb([250, 100, 7]));
        let batched = processor
            .preprocess(&[a.clone(), b.clone()], &config)
            .unwrap();
        let single_a = processor.preprocess(&[a], &config).unwrap();
        let single_b = processor.preprocess(&[b], &config).unwrap();
        let rows_a = single_a.encoder_input.shape()[0];
        let rows_b = single_b.encoder_input.shape()[0];
        assert_eq!(batched.encoder_input.shape()[0], rows_a + rows_b);
        assert_eq!(
            batched.encoder_input.slice_axis0(0, rows_a).unwrap(),
            single_a.encoder_input
        );
        assert_eq!(
            batched.encoder_input.slice_axis0(rows_a, rows_b).unwrap(),
            single_b.encoder_input
        );
        assert_eq!(
            batched.feature_token_counts,
            vec![
                single_a.feature_token_counts[0],
                single_b.feature_token_counts[0]
            ]
        );
        assert_eq!(
            processor.calculate_num_tokens(100, 60, &config).unwrap(),
            single_a.feature_token_counts[0]
        );
    }

    #[test]
    fn video_frames_pair_up_and_the_last_frame_fills_an_odd_group() {
        let processor = MuseSparkVisionProcessor::new();
        let config = video_config();
        let frames: Vec<DynamicImage> = (0..3u8)
            .map(|value| solid(56, 28, Rgb([value * 50, 0, 0])))
            .collect();
        let output = processor.preprocess_video(&frames, &config).unwrap();
        // 56x28 at patch 14 is a 2x4 grid: 8 rows per group, two groups.
        assert_eq!(output.encoder_input.shape(), &[16, 1176]);
        assert_eq!(output.feature_token_counts, vec![4]);
        assert!(matches!(
            output.model_specific.get("video_grid_thw"),
            Some(ModelSpecificValue::IntTensor { data, shape }) if data == &[2, 2, 4] && shape == &[1, 3]
        ));
        assert!(matches!(
            output.model_specific.get("patches_per_video"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &[16]
        ));
        let values = output.encoder_input.as_f32().into_owned();
        let norm = |byte: f32| (byte - 127.5) / 127.5;
        // Group 0: frames 0 and 1 in the two temporal slots.
        assert_eq!(values[[0, 0]], norm(0.0));
        assert_eq!(values[[0, 588]], norm(50.0));
        // Group 1: frame 2 repeated.
        assert_eq!(values[[8, 0]], norm(100.0));
        assert_eq!(values[[8, 588]], norm(100.0));

        let rgb: Vec<RgbImage> = frames.iter().map(DynamicImage::to_rgb8).collect();
        let refs: Vec<RgbFrameRef<'_>> = rgb
            .iter()
            .map(|frame| RgbFrameRef {
                width: frame.width(),
                height: frame.height(),
                data: frame.as_raw(),
            })
            .collect();
        let from_rgb = processor.preprocess_video_rgb(&refs, &config).unwrap();
        assert_eq!(from_rgb.encoder_input, output.encoder_input);
        assert_eq!(from_rgb.feature_token_counts, output.feature_token_counts);
    }

    #[test]
    fn video_frames_resize_under_the_per_frame_cap() {
        let config = video_config();
        // 1280x720 under 256 tokens per frame: the reference grid is 12x21.
        let plan = MuseSparkVisionProcessor::video_plan(1280, 720, &config).unwrap();
        assert_eq!((plan.grid_h, plan.grid_w, plan.tokens), (24, 42, 252));
        assert_eq!((plan.width, plan.height), (588, 336));
        // The same frame as an image gets the image cap instead.
        let plan = MuseSparkVisionProcessor::image_plan(1280, 720, &image_config()).unwrap();
        assert_eq!((plan.grid_h, plan.grid_w, plan.tokens), (52, 92, 1196));
    }

    #[test]
    fn empty_and_ragged_inputs_are_errors() {
        let processor = MuseSparkVisionProcessor::new();
        assert!(matches!(
            processor.preprocess(&[], &image_config()),
            Err(TransformError::EmptyBatch)
        ));
        assert!(matches!(
            processor.preprocess_video(&[], &video_config()),
            Err(TransformError::EmptyBatch)
        ));
        let frames = [solid(56, 28, Rgb([0; 3])), solid(28, 28, Rgb([0; 3]))];
        assert!(matches!(
            processor.preprocess_video(&frames, &video_config()),
            Err(TransformError::InconsistentShapes)
        ));
    }
}
