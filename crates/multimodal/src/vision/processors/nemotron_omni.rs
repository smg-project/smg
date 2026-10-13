//! Nemotron-H Omni (C-RADIO) image preprocessing.
//!
//! Port of the serving engine's dynamic-resolution Nemotron-Omni image path
//! (`DynamicResolutionImageTiler`). Each image keeps its aspect ratio and is
//! resized to an even grid of 16-pixel patches with between `min_num_patches`
//! and `max_num_patches` patches, using torch's antialiased bicubic on the
//! image as floats; it is then normalized with the CLIP mean and std. A
//! transparent image is first composited onto white, as the engine's image loader
//! does; the decoder also reports a truecolor PNG's tRNS colour key as alpha,
//! which the engine leaves opaque. The engine receives `(patches, 3 * p * p)` rows (each patch flattened channel,
//! row, column; patches row-major) and `image_grid_hw`, and its 2x2 pixel
//! shuffle makes `h * w / 4` tokens of an image.
//!
//! A video clip's frames all keep the clip's aspect ratio on an even grid of
//! about 1024 patches, are resized and normalized the same way, and are sent
//! as `(frames * h * w, 3 * p * p)` rows with `video_grid_thw` `(frames, h, w)`;
//! the engine pairs frames into tubelets of `h * w / 4` tokens each.
//!
//! The engine also scales a request's images down to share `4 * (max_model_len -
//! prompt_len - 4)` patches. This port does not: each image gets the
//! per-image maximum, so a request whose images overflow the engine's context
//! is refused where the engine would shrink them.

use std::borrow::Cow;

use image::DynamicImage;
use ndarray::Array2;
use serde_json::{json, Value};

use crate::{
    encoder_inputs::ModelSpecificValue,
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::{resize_bicubic_torch_aa_f32, rgb_bytes, TransformError},
    },
};

const PATCH_SIZE: usize = 16;
/// The checkpoint's `vision_config.args.{min,max}_num_patches`.
const MIN_NUM_PATCHES: usize = 1024;
const MAX_NUM_PATCHES: usize = 13312;
/// The checkpoint's `vision_config.video_target_num_patches`.
const VIDEO_TARGET_PATCHES: usize = 1024;
/// Frames per tubelet (`vision_config.video_temporal_patch_size`).
pub(crate) const VIDEO_TEMPORAL_PATCH_SIZE: usize = 2;
/// The 2x2 pixel shuffle needs an even patch grid.
const GRID_DIVISOR: usize = 2;

/// Embeddings of a `rows x cols` patch grid after the 2x2 pixel shuffle.
pub(crate) fn grid_tokens(rows: usize, cols: usize) -> usize {
    rows * cols / (GRID_DIVISOR * GRID_DIVISOR)
}
const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Whether a checkpoint config value equals the one this port implements,
/// comparing numbers by value (`16` and `16.0` are the same).
fn same_config_value(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_config_value(a, b))
        }
        _ => actual == expected,
    }
}

/// C-RADIO dynamic-resolution image preprocessor for Nemotron-H Omni.
#[derive(Debug, Clone, Default)]
pub struct NemotronOmniProcessor;

impl NemotronOmniProcessor {
    pub fn new() -> Self {
        Self
    }

    /// The checkpoint config field, as a JSON pointer, whose value differs from
    /// the geometry this port implements, if any.
    pub fn mismatched_config_field(config: &Value) -> Option<&'static str> {
        [
            ("/patch_size", json!(PATCH_SIZE)),
            ("/downsample_ratio", json!(0.5)),
            ("/norm_mean", json!(CLIP_MEAN)),
            ("/norm_std", json!(CLIP_STD)),
            (
                "/vision_config/args/min_num_patches",
                json!(MIN_NUM_PATCHES),
            ),
            (
                "/vision_config/args/max_num_patches",
                json!(MAX_NUM_PATCHES),
            ),
            (
                "/vision_config/video_target_num_patches",
                json!(VIDEO_TARGET_PATCHES),
            ),
            (
                "/vision_config/video_temporal_patch_size",
                json!(VIDEO_TEMPORAL_PATCH_SIZE),
            ),
            ("/vision_config/video_maintain_aspect_ratio", json!(true)),
        ]
        .into_iter()
        .find(|(pointer, value)| {
            !config
                .pointer(pointer)
                .is_some_and(|actual| same_config_value(actual, value))
        })
        .map(|(pointer, _)| pointer)
        .or_else(|| {
            // The engine sizes video frames by exactly one of these two.
            let size = "/vision_config/video_target_img_size";
            config
                .pointer(size)
                .is_some_and(|v| !v.is_null())
                .then_some(size)
        })
    }

    /// The `(rows, cols)` patch grid an image is resized to, as the engine's
    /// `process_media` plans it at the per-image maximum; `None` for a strip so
    /// thin it rounds to no patches, which the engine fails on too.
    pub fn plan_grid(width: u32, height: u32) -> Option<(usize, usize)> {
        let budget = MAX_NUM_PATCHES;
        let p = PATCH_SIZE as f64;
        let closest_h = (f64::from(height) / p + 0.5).round_ties_even() as usize;
        let closest_w = (f64::from(width) / p + 0.5).round_ties_even() as usize;
        let factor = (budget as f64 / (closest_h * closest_w) as f64)
            .sqrt()
            .min(1.0);
        let mut rows = (factor * closest_h as f64).floor() as usize;
        let mut cols = (factor * closest_w as f64).floor() as usize;
        if rows * cols == 0 {
            return None;
        }
        if rows * cols < MIN_NUM_PATCHES {
            let up = (MIN_NUM_PATCHES as f64 / (rows * cols) as f64).sqrt();
            rows = (up * rows as f64).ceil() as usize;
            cols = (up * cols as f64).ceil() as usize;
        }
        if !rows.is_multiple_of(GRID_DIVISOR) {
            let rem = rows % GRID_DIVISOR;
            rows = if (rows + GRID_DIVISOR - rem) * cols <= budget {
                rows + GRID_DIVISOR - rem
            } else {
                GRID_DIVISOR.max(rows - rem)
            };
        }
        if !cols.is_multiple_of(GRID_DIVISOR) {
            let rem = cols % GRID_DIVISOR;
            cols = if rows * (cols + GRID_DIVISOR - rem) <= budget {
                cols + GRID_DIVISOR - rem
            } else {
                GRID_DIVISOR.max(cols - rem)
            };
        }
        Some((rows, cols))
    }

    /// The `(rows, cols)` grid of a video's frames, as the engine's
    /// `_compute_aspect_preserving_size` plans it.
    pub fn plan_video_grid(width: u32, height: u32) -> (usize, usize) {
        let target = VIDEO_TARGET_PATCHES as f64;
        let aspect = f64::from(width) / f64::from(height.max(1));
        let rows = ((target / aspect).sqrt().round_ties_even() as usize).max(1);
        let cols = ((target * aspect).sqrt().round_ties_even() as usize).max(1);
        let (rows_up, cols_up) = (rows.next_multiple_of(2), cols.next_multiple_of(2));
        if rows_up * cols_up <= VIDEO_TARGET_PATCHES {
            (rows_up, cols_up)
        } else {
            (
                GRID_DIVISOR.max(rows - rows % 2),
                GRID_DIVISOR.max(cols - cols % 2),
            )
        }
    }

    /// The image's RGB bytes with any transparency composited onto white the
    /// way the engine's loader does it: PIL's `paste` with the alpha as mask.
    fn rgb_on_white(image: &DynamicImage) -> (usize, usize, Cow<'_, [u8]>) {
        if !image.color().has_alpha() {
            return rgb_bytes(image);
        }
        let rgba = image.to_rgba8();
        let mut out = Vec::with_capacity(rgba.as_raw().len() / 4 * 3);
        for px in rgba.as_raw().as_chunks::<4>().0 {
            let alpha = u32::from(px[3]);
            for &c in &px[..3] {
                // PIL's BLEND: (255 * (255 - a) + c * a) / 255, rounded with its DIV255.
                let v = 255 * (255 - alpha) + u32::from(c) * alpha + 128;
                out.push(((v + (v >> 8)) >> 8) as u8);
            }
        }
        (
            rgba.width() as usize,
            rgba.height() as usize,
            Cow::Owned(out),
        )
    }

    /// Resize, normalize and patchify one RGB image: `rows * cols * 3 * p * p` values.
    fn patches(
        raw: &[u8],
        width: u32,
        height: u32,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, TransformError> {
        let p = PATCH_SIZE;
        let (out_w, out_h) = (cols * p, rows * p);
        let resized = resize_bicubic_torch_aa_f32(raw, width, height, out_w as u32, out_h as u32)?;
        let scale = CLIP_STD.map(|s| s as f32);
        let mean = CLIP_MEAN.map(|m| m as f32);
        let patch_len = 3 * p * p;
        let mut out = vec![0.0_f32; rows * cols * patch_len];
        for i in 0..rows {
            for j in 0..cols {
                let patch = &mut out[(i * cols + j) * patch_len..][..patch_len];
                for c in 0..3 {
                    for dy in 0..p {
                        for dx in 0..p {
                            let v = resized[((i * p + dy) * out_w + j * p + dx) * 3 + c];
                            // f32 op order matches the reference: x / 255, - mean, / std.
                            patch[(c * p + dy) * p + dx] = (v / 255.0 - mean[c]) / scale[c];
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// One clip's frames as patches plus its `video_grid_thw`.
    #[expect(
        clippy::expect_used,
        reason = "the patch buffer is sized from the grid it is reshaped by"
    )]
    fn video_patches(
        frames: &[RgbFrameRef<'_>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let first = frames.first().ok_or(TransformError::EmptyBatch)?;
        let (rows, cols) = Self::plan_video_grid(first.width, first.height);
        let patch_len = 3 * PATCH_SIZE * PATCH_SIZE;
        let mut encoder_input = Vec::with_capacity(frames.len() * rows * cols * patch_len);
        for frame in frames {
            if (frame.width, frame.height) != (first.width, first.height) {
                return Err(TransformError::ShapeError(format!(
                    "Nemotron-H Omni video frames must share one size, got {}x{} and {}x{}",
                    first.width, first.height, frame.width, frame.height
                )));
            }
            encoder_input.extend(Self::patches(
                frame.data,
                frame.width,
                frame.height,
                rows,
                cols,
            )?);
        }
        let tubelets = frames.len().div_ceil(VIDEO_TEMPORAL_PATCH_SIZE);
        let encoder_input =
            Array2::from_shape_vec((frames.len() * rows * cols, patch_len), encoder_input)
                .expect("frame patches match their shape by construction");
        Ok(PreprocessedEncoderInputs::new(
            encoder_input,
            vec![tubelets * grid_tokens(rows, cols)],
            vec![(first.width, first.height)],
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::IntTensor {
                data: vec![frames.len() as i64, rows as i64, cols as i64],
                shape: vec![1, 3],
            },
        )
        .with_extra(
            "patches_per_video",
            ModelSpecificValue::int_1d(vec![(frames.len() * rows * cols) as i64]),
        ))
    }
}

impl VisionPreProcessor for NemotronOmniProcessor {
    fn supports_per_image_preprocessing(&self) -> bool {
        true
    }

    fn default_mean(&self) -> [f64; 3] {
        CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        CLIP_STD
    }

    #[expect(
        clippy::expect_used,
        reason = "per-image patch arrays are standard-layout and the concatenation is sized by construction"
    )]
    fn preprocess(
        &self,
        images: &[DynamicImage],
        _config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let patch_len = 3 * PATCH_SIZE * PATCH_SIZE;
        let mut encoder_input = Vec::new();
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len() * 2);
        let mut patches_per_image = Vec::with_capacity(images.len());
        for image in images {
            let (rows, cols) = Self::plan_grid(image.width(), image.height()).ok_or_else(|| {
                TransformError::ShapeError(format!(
                    "a {}x{} image is too thin for Nemotron-H Omni's patch grid",
                    image.width(),
                    image.height()
                ))
            })?;
            let (width, height, raw) = Self::rgb_on_white(image);
            encoder_input.extend(Self::patches(
                &raw,
                width as u32,
                height as u32,
                rows,
                cols,
            )?);
            feature_token_counts.push(grid_tokens(rows, cols));
            item_sizes.push((image.width(), image.height()));
            grids.extend([rows as i64, cols as i64]);
            patches_per_image.push((rows * cols) as i64);
        }
        let total_patches = encoder_input.len() / patch_len;
        let encoder_input = Array2::from_shape_vec((total_patches, patch_len), encoder_input)
            .expect("concatenated patch buffer matches its shape by construction");
        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra(
                    "image_grid_hw",
                    ModelSpecificValue::IntTensor {
                        data: grids,
                        shape: vec![images.len(), 2],
                    },
                )
                .with_extra(
                    "patches_per_image",
                    ModelSpecificValue::int_1d(patches_per_image),
                ),
        )
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        _config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let rgb: Vec<_> = frames.iter().map(rgb_bytes).collect();
        let refs: Vec<_> = rgb
            .iter()
            .map(|(width, height, raw)| RgbFrameRef {
                width: *width as u32,
                height: *height as u32,
                data: raw,
            })
            .collect();
        Self::video_patches(&refs)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        _config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Self::video_patches(frames)
    }

    fn calculate_num_tokens(
        &self,
        width: u32,
        height: u32,
        _config: &PreProcessorConfig,
    ) -> Result<usize, TransformError> {
        let (rows, cols) = Self::plan_grid(width, height).ok_or_else(|| {
            TransformError::ShapeError(format!(
                "a {width}x{height} image is too thin for Nemotron-H Omni's patch grid"
            ))
        })?;
        Ok(grid_tokens(rows, cols))
    }

    fn model_name(&self) -> &'static str {
        "nemotron_omni"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Token counts the engine's tiler gives these sizes at the per-image maximum.
    #[test]
    fn plans_the_reference_grids() {
        let tokens = |w, h| {
            let (rows, cols) = NemotronOmniProcessor::plan_grid(w, h).unwrap();
            rows * cols / 4
        };
        assert_eq!(tokens(224, 224), 256);
        assert_eq!(tokens(640, 480), 300);
        assert_eq!(tokens(1280, 720), 920);
        assert_eq!(tokens(1920, 1080), 2040);
        assert_eq!(tokens(2048, 2048), 3306);
        assert_eq!(tokens(4000, 3000), 3300);
        // Rounds to a zero-height grid; the engine divides by its area and fails.
        assert_eq!(NemotronOmniProcessor::plan_grid(300_000, 8), None);
        // The count refuses what the grid planner refuses, as the
        // preprocessing does.
        let error = NemotronOmniProcessor::new()
            .calculate_num_tokens(300_000, 8, &PreProcessorConfig::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("too thin"), "{error}");
    }

    /// Frame grids the engine's `_compute_aspect_preserving_size` gives these sizes.
    #[test]
    fn plans_the_reference_video_grids() {
        let grid = NemotronOmniProcessor::plan_video_grid;
        assert_eq!(grid(1920, 1080), (24, 42));
        assert_eq!(grid(1280, 720), (24, 42));
        assert_eq!(grid(640, 480), (28, 36));
        assert_eq!(grid(512, 512), (32, 32));
        // Square roots within 1e-9 of a half, which Python rounds to even.
        assert_eq!(grid(4096, 841), (14, 70));
        assert_eq!(grid(4096, 3969), (30, 32));
        // An odd row count rounds up to an even grid within the target.
        assert_eq!(grid(16, 159), (102, 10));
    }

    #[test]
    fn composites_transparency_onto_white_as_pil_does() {
        let image = DynamicImage::from(
            image::RgbaImage::from_raw(
                4,
                1,
                vec![0, 0, 0, 0, 200, 10, 99, 128, 37, 250, 1, 1, 9, 8, 7, 255],
            )
            .unwrap(),
        );
        let (_, _, rgb) = NemotronOmniProcessor::rgb_on_white(&image);
        // PIL: Image.new("RGB", size, white).paste(image, mask=alpha).
        assert_eq!(
            rgb.as_ref(),
            [255, 255, 255, 227, 132, 177, 254, 255, 254, 9, 8, 7]
        );
    }

    #[test]
    fn keeps_video_frames_in_order() {
        let frames: Vec<Vec<u8>> = [10_u8, 120, 240]
            .iter()
            .map(|&v| vec![v; 32 * 32 * 3])
            .collect();
        let refs: Vec<_> = frames
            .iter()
            .map(|data| RgbFrameRef {
                width: 32,
                height: 32,
                data,
            })
            .collect();
        let out = NemotronOmniProcessor::video_patches(&refs).unwrap();
        // Two tubelets (the third frame pairs with itself) of a 32x32 grid.
        assert_eq!(out.feature_token_counts, [2 * 32 * 32 / 4]);
        let values = out.encoder_input.as_f32();
        let rows_per_frame = values.shape()[0] / 3;
        let first_value = |frame: usize| values[[frame * rows_per_frame, 0]];
        assert!(first_value(0) < first_value(1) && first_value(1) < first_value(2));
    }
}
