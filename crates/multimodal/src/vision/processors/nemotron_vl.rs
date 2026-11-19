//! Nemotron VL / Omni (C-RADIO) image and video preprocessing.
//!
//! Port of the serving engine's `NanoNemotronVLProcessor`, which serves the
//! family's checkpoints, tiled and dynamic-resolution, from their model config ([`NemotronVLGeometry`]):
//!
//! - **Tiled images** (`vision_config.args` without `min_num_patches`): the
//!   InternVL `dynamic_preprocess`. The image is resized to the tile grid
//!   whose aspect is closest to its own (at most 12 tiles of
//!   `force_image_size` pixels), cut into tiles row by row, and followed by a
//!   thumbnail of the whole image when `use_thumbnail` and there is more than
//!   one tile. The engine receives `pixel_values_flat` `(tiles, 3, S, S)` with
//!   `image_num_patches`, `S * S / p^2 * r^2` tokens per tile.
//! - **Dynamic resolution** (`min_num_patches` set): the engine's
//!   `DynamicResolutionImageTiler`. Each image keeps its aspect on an even
//!   grid of `p`-pixel patches, between `min_num_patches` and
//!   `max_num_patches`, the request's images sharing `shuffle^2 *
//!   (token_budget - 4)` patches when the caller passes the context left for
//!   them as [`token_budget`](VisionPreprocessingContext::token_budget). One
//!   `(3, H, W)` `pixel_values_flat` per image, with `imgs_sizes` and
//!   `num_tokens_per_image`; the engine's pixel shuffle makes `h * w /
//!   shuffle^2` tokens of a `h x w` grid.
//! - **Video frames** are resized to about `video_target_num_patches` patches
//!   (or `ceil(video_target_img_size / p)^2`), keeping the clip's aspect when
//!   `video_maintain_aspect_ratio`, else to `force_image_size` squares, and
//!   sent as `pixel_values_flat_video` `(frames, 3, H, W)` with
//!   `video_num_patches`, the sampled `frames_indices` and
//!   `frame_duration_ms`; the engine pairs frames into tubelets of
//!   `video_temporal_patch_size`.
//!
//! The grid and budget planners follow
//! [`nemotron_omni`](super::nemotron_omni)'s, with the checkpoint's patch
//! size, patch counts and shuffle factor in place of that module's constants.
//!
//! Every resize is torch's antialiased bicubic on the image as floats,
//! unrounded, then `(x / 255 - mean) / std`. A transparent image is first
//! composited onto white, as the engine's image loader does.

use std::collections::HashMap;

use image::DynamicImage;
use ndarray::{ArrayD, IxDyn};
use serde_json::Value;

use crate::{
    encoder_inputs::ModelSpecificValue,
    types::{FieldLayout, RgbFrameRef, VideoSamplingInfo},
    vision::{
        processor::{
            PreprocessedEncoderInputs, VideoClipOptions, VisionPreProcessor,
            VisionPreprocessingContext,
        },
        processors::nemotron_omni::{
            NemotronOmniProcessor, BUDGET_RESERVED_TOKENS, MAX_BUDGET_ROUNDS,
        },
        transforms::{resize_bicubic_torch_aa_f32, TransformError},
    },
};

/// The engine's tile cap (`DEFAULT_NUM_TILES`).
const MAX_NUM_TILES: usize = 12;
/// A dynamic-resolution image's tensors beside its pixels, one entry per image.
pub(crate) const DYNAMIC_IMAGE_SIDE_TENSORS: [&str; 2] = ["imgs_sizes", "num_tokens_per_image"];

/// A geometry field the model config lacks, or holds with a value the engine
/// would refuse; each names the field's JSON pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronVLConfigError {
    /// The field is absent or null.
    Missing(&'static str),
    /// The field holds a value the engine's processor refuses.
    Invalid(&'static str),
}

/// How video frames are sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFrameSize {
    /// `force_image_size` squares, whatever the clip's aspect.
    Square,
    /// About `patches` patches: the clip's aspect on an even grid when
    /// `keep_aspect`, else the largest even square within.
    Patches { patches: usize, keep_aspect: bool },
}

/// The checkpoint fields the engine's processor reads.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronVLGeometry {
    pub patch_size: usize,
    pub downsample_ratio: f64,
    /// `force_image_size`: the tile, thumbnail and square video frame side.
    pub image_size: usize,
    pub use_thumbnail: bool,
    pub norm_mean: [f64; 3],
    pub norm_std: [f64; 3],
    /// `vision_config.args.{min,max}_num_patches` for dynamic resolution;
    /// `None` tiles. A non-positive maximum is no maximum.
    pub dynamic: Option<(usize, usize)>,
    pub video_frame_size: VideoFrameSize,
    pub video_temporal_patch_size: usize,
}

impl NemotronVLGeometry {
    /// Read the geometry from a model config.
    pub fn from_model_config(config: &Value) -> Result<Self, NemotronVLConfigError> {
        use NemotronVLConfigError::{Invalid, Missing};
        let field = |ptr: &'static str| config.pointer(ptr).filter(|v| !v.is_null());
        let number = |ptr| field(ptr).ok_or(Missing(ptr))?.as_f64().ok_or(Invalid(ptr));
        let count = |ptr| {
            number(ptr).and_then(|v| {
                (v >= 1.0 && v.fract() == 0.0)
                    .then_some(v as usize)
                    .ok_or(Invalid(ptr))
            })
        };
        let optional_count = |ptr| field(ptr).map(|_| count(ptr)).transpose();
        let flag = |ptr| field(ptr).and_then(Value::as_bool).unwrap_or(false);
        let triple = |ptr| {
            field(ptr)
                .ok_or(Missing(ptr))?
                .as_array()
                .and_then(|values| <[&Value; 3]>::try_from(values.iter().collect::<Vec<_>>()).ok())
                .and_then(|values| {
                    let [a, b, c] = values.map(Value::as_f64);
                    Some([a?, b?, c?])
                })
                .ok_or(Invalid(ptr))
        };

        let patch_size = count("/patch_size")?;
        let downsample_ratio = number("/downsample_ratio")?;
        if !(downsample_ratio > 0.0 && downsample_ratio <= 1.0) {
            return Err(Invalid("/downsample_ratio"));
        }
        let dynamic = match field("/vision_config/args/min_num_patches") {
            None => None,
            Some(_) => {
                let max = number("/vision_config/args/max_num_patches")?;
                let max = if max <= 0.0 { usize::MAX } else { max as usize };
                Some((count("/vision_config/args/min_num_patches")?, max))
            }
        };
        let target_patches = optional_count("/vision_config/video_target_num_patches")?;
        let target_size = optional_count("/vision_config/video_target_img_size")?;
        let video_frame_size = match (target_patches, target_size) {
            (Some(_), Some(_)) => return Err(Invalid("/vision_config/video_target_img_size")),
            (Some(patches), None) => Some(patches),
            (None, Some(size)) => Some(size.div_ceil(patch_size).pow(2)),
            (None, None) => None,
        }
        .map_or(VideoFrameSize::Square, |patches| VideoFrameSize::Patches {
            patches,
            keep_aspect: flag("/vision_config/video_maintain_aspect_ratio"),
        });
        Ok(Self {
            patch_size,
            downsample_ratio,
            image_size: count("/force_image_size")?,
            use_thumbnail: flag("/use_thumbnail"),
            norm_mean: triple("/norm_mean")?,
            norm_std: triple("/norm_std")?,
            dynamic,
            video_frame_size,
            video_temporal_patch_size: optional_count("/vision_config/video_temporal_patch_size")?
                .unwrap_or(1),
        })
    }

    /// The pixel-shuffle factor per side, `round(1 / downsample_ratio)`; the
    /// patch grids are kept multiples of it.
    pub fn shuffle(&self) -> usize {
        (1.0 / self.downsample_ratio).round() as usize
    }

    /// Tokens of one `force_image_size` tile or square frame
    /// (`num_image_token`).
    pub fn tile_tokens(&self) -> usize {
        let side = self.image_size / self.patch_size;
        ((side * side) as f64 * self.downsample_ratio * self.downsample_ratio) as usize
    }

    /// Tokens of one video frame (one tubelet) of `rows x cols` patches.
    pub fn frame_tokens(&self, rows: usize, cols: usize) -> usize {
        match self.video_frame_size {
            VideoFrameSize::Square => self.tile_tokens(),
            VideoFrameSize::Patches { .. } => {
                (rows as f64 * self.downsample_ratio) as usize
                    * (cols as f64 * self.downsample_ratio) as usize
            }
        }
    }

    /// Tubelets of a clip of `frames` frames; a short last tubelet is padded
    /// by the engine.
    pub fn tubelets(&self, frames: usize) -> usize {
        frames.div_ceil(self.video_temporal_patch_size.max(1))
    }
}

/// The engine's tile grid candidates `(columns, rows)`, by area
/// (`get_internvl_target_ratios`).
fn target_ratios() -> Vec<(usize, usize)> {
    let mut ratios: Vec<_> = (1..=MAX_NUM_TILES)
        .flat_map(|columns| (1..=MAX_NUM_TILES).map(move |rows| (columns, rows)))
        .filter(|(columns, rows)| columns * rows <= MAX_NUM_TILES)
        .collect();
    ratios.sort_by_key(|(columns, rows)| columns * rows);
    ratios
}

/// C-RADIO preprocessor for the Nemotron VL / Omni family.
#[derive(Debug, Clone)]
pub struct NemotronVLProcessor {
    geometry: NemotronVLGeometry,
}

impl NemotronVLProcessor {
    pub fn new(geometry: NemotronVLGeometry) -> Self {
        Self { geometry }
    }

    /// The `(rows, cols)` patch grid an image is resized to at the per-image
    /// maximum; `None` for a strip so thin it rounds to no patches, which the
    /// engine fails on too, or for a tiling geometry.
    pub fn plan_grid(&self, width: u32, height: u32) -> Option<(usize, usize)> {
        let (_, max) = self.geometry.dynamic?;
        self.plan_grid_within(width, height, max)
    }

    /// The engine's `process_media`: the grid of an image within `budget`
    /// patches. The minimum is enforced only above it: an image whose budget
    /// is the minimum keeps the grid the budget scales it to, however small.
    fn plan_grid_within(&self, width: u32, height: u32, budget: usize) -> Option<(usize, usize)> {
        let (min, _) = self.geometry.dynamic?;
        let divisor = self.geometry.shuffle();
        let p = self.geometry.patch_size as f64;
        // Python's round() of `x + 0.5`: ties go to even.
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
        if budget > min && rows * cols < min {
            let up = (min as f64 / (rows * cols) as f64).sqrt();
            rows = (up * rows as f64).ceil() as usize;
            cols = (up * cols as f64).ceil() as usize;
        }
        if !rows.is_multiple_of(divisor) {
            let rem = rows % divisor;
            rows = if (rows + divisor - rem) * cols <= budget {
                rows + divisor - rem
            } else {
                divisor.max(rows - rem)
            };
        }
        if !cols.is_multiple_of(divisor) {
            let rem = cols % divisor;
            cols = if rows * (cols + divisor - rem) <= budget {
                cols + divisor - rem
            } else {
                divisor.max(cols - rem)
            };
        }
        Some((rows, cols))
    }

    /// The grids of one call's images, as the engine's `compute_params` plans
    /// them: without a token budget, each at the per-image maximum; with one,
    /// sharing `shuffle^2 * (token_budget - 4)` patches, each image's share
    /// shrunk in proportion to its patches until the grids fit. A budget
    /// below every image's minimum is raised to it.
    fn plan_grids(
        &self,
        sizes: &[(u32, u32)],
        token_budget: Option<usize>,
    ) -> Result<Vec<(usize, usize)>, TransformError> {
        let (min, max) = self.geometry.dynamic.ok_or_else(|| {
            TransformError::ShapeError("a tiling geometry plans no dynamic grids".to_string())
        })?;
        let grid = |&(width, height): &(u32, u32), budget| {
            self.plan_grid_within(width, height, budget).ok_or_else(|| {
                TransformError::ShapeError(format!(
                    "a {width}x{height} image is too thin for Nemotron's patch grid"
                ))
            })
        };
        let Some(token_budget) = token_budget else {
            return sizes.iter().map(|size| grid(size, max)).collect();
        };
        let available = token_budget
            .saturating_sub(BUDGET_RESERVED_TOKENS)
            .saturating_mul(self.geometry.shuffle().pow(2))
            .max(min * sizes.len());
        let mut budgets = vec![available.clamp(min, max.max(min)); sizes.len()];
        for _ in 0..MAX_BUDGET_ROUNDS {
            let grids = sizes
                .iter()
                .zip(&budgets)
                .map(|(size, &budget)| grid(size, budget))
                .collect::<Result<Vec<_>, _>>()?;
            let used: usize = grids.iter().map(|(rows, cols)| rows * cols).sum();
            if used <= available {
                return Ok(grids);
            }
            let scale = available as f64 / used as f64;
            let scaled: Vec<usize> = grids
                .iter()
                .map(|(rows, cols)| (((rows * cols) as f64 * scale) as usize).max(min))
                .collect();
            // Shares that no longer shrink are stuck at the minimum.
            budgets = if scaled.iter().zip(&budgets).any(|(new, old)| new < old) {
                scaled
            } else {
                vec![min; sizes.len()]
            };
        }
        Err(TransformError::ShapeError(format!(
            "Nemotron's images did not fit a {token_budget}-token budget in \
             {MAX_BUDGET_ROUNDS} rounds"
        )))
    }

    /// The `(columns, rows)` tile grid of an image (`find_closest_aspect_ratio`):
    /// the candidate whose aspect is closest to the image's, a later equally
    /// close one when the image covers more than half its tiles.
    pub fn plan_tiles(&self, width: u32, height: u32) -> (usize, usize) {
        let aspect = f64::from(width) / f64::from(height);
        let area = f64::from(width) * f64::from(height);
        let side = self.geometry.image_size as f64;
        let mut best = (f64::INFINITY, (1, 1));
        for (columns, rows) in target_ratios() {
            let diff = (aspect - columns as f64 / rows as f64).abs();
            if diff < best.0
                || (diff == best.0 && area > 0.5 * side * side * (columns * rows) as f64)
            {
                best = (diff, (columns, rows));
            }
        }
        best.1
    }

    /// Tiles of an image, its thumbnail included.
    fn tile_count(&self, columns: usize, rows: usize) -> usize {
        let tiles = columns * rows;
        tiles + usize::from(self.geometry.use_thumbnail && tiles != 1)
    }

    /// The `(height, width)` in pixels a clip's frames are resized to
    /// (`get_video_target_size_and_feature_size`, else `force_image_size`).
    pub fn plan_video_size(&self, width: u32, height: u32) -> (usize, usize) {
        let p = self.geometry.patch_size;
        let divisor = self.geometry.shuffle();
        match self.geometry.video_frame_size {
            VideoFrameSize::Square => (self.geometry.image_size, self.geometry.image_size),
            VideoFrameSize::Patches {
                patches,
                keep_aspect: false,
            } => {
                let side = divisor.max(((patches as f64).sqrt() as usize / 2) * 2);
                (side * p, side * p)
            }
            VideoFrameSize::Patches {
                patches,
                keep_aspect: true,
            } => {
                let target = patches as f64;
                let aspect = f64::from(width) / f64::from(height.max(1));
                let rows = ((target / aspect).sqrt().round_ties_even() as usize).max(1);
                let cols = ((target * aspect).sqrt().round_ties_even() as usize).max(1);
                let (rows_up, cols_up) = (
                    rows.next_multiple_of(divisor),
                    cols.next_multiple_of(divisor),
                );
                let (rows, cols) = if rows_up * cols_up <= patches {
                    (rows_up, cols_up)
                } else {
                    (
                        divisor.max(rows - rows % divisor),
                        divisor.max(cols - cols % divisor),
                    )
                };
                (rows * p, cols * p)
            }
        }
    }

    /// `raw` (`width x height` RGB) resized to `out_w x out_h` as the engine's
    /// `_bicubic_resize_and_normalize` resizes it: interleaved floats on the
    /// byte scale, the bytes themselves when the size does not change.
    fn resized(
        raw: &[u8],
        (width, height): (usize, usize),
        (out_w, out_h): (usize, usize),
    ) -> Result<Vec<f32>, TransformError> {
        if width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(raw.len()) {
            return Err(TransformError::ShapeError(format!(
                "a {width}x{height} RGB frame needs {width}*{height}*3 bytes, got {}",
                raw.len()
            )));
        }
        if (width, height) == (out_w, out_h) {
            return Ok(raw.iter().copied().map(f32::from).collect());
        }
        let (w, h, ow, oh) = (width as u32, height as u32, out_w as u32, out_h as u32);
        resize_bicubic_torch_aa_f32(raw, w, h, ow, oh)
    }

    /// Append the `(x, y, w, h)` window of a resized image `stride` pixels
    /// wide, channel-major, as `(x / 255 - mean) / std`.
    fn push_window(
        &self,
        out: &mut Vec<f32>,
        resized: &[f32],
        stride: usize,
        (x0, y0, w, h): (usize, usize, usize, usize),
    ) {
        let mean = self.geometry.norm_mean.map(|m| m as f32);
        let std = self.geometry.norm_std.map(|s| s as f32);
        let (pixels, _) = resized.as_chunks::<3>();
        for c in 0..3 {
            let (mean, std) = (mean[c], std[c]);
            for y in y0..y0 + h {
                let row = &pixels[y * stride + x0..][..w];
                out.extend(row.iter().map(|px| (px[c] / 255.0 - mean) / std));
            }
        }
    }

    fn encoder_input(data: Vec<f32>, shape: [usize; 4]) -> Result<ArrayD<f32>, TransformError> {
        ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| TransformError::ShapeError(format!("{shape:?}: {e}")))
    }

    /// One batch of a call's tiled images.
    fn preprocess_tiled(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let side = self.geometry.image_size;
        let plans: Vec<_> = images
            .iter()
            .map(|image| self.plan_tiles(image.width(), image.height()))
            .collect();
        let total_tiles: usize = plans
            .iter()
            .map(|&(columns, rows)| self.tile_count(columns, rows))
            .sum();
        let mut pixels = Vec::with_capacity(total_tiles * 3 * side * side);
        let mut tiles_per_image = Vec::with_capacity(images.len());
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        for (image, &(columns, rows)) in images.iter().zip(&plans) {
            let (width, height, raw) = NemotronOmniProcessor::rgb_on_white(image);
            let (out_w, out_h) = (columns * side, rows * side);
            {
                let resized = Self::resized(&raw, (width, height), (out_w, out_h))?;
                for row in 0..rows {
                    for column in 0..columns {
                        self.push_window(
                            &mut pixels,
                            &resized,
                            out_w,
                            (column * side, row * side, side, side),
                        );
                    }
                }
            }
            let tiles = self.tile_count(columns, rows);
            if tiles > columns * rows {
                let thumbnail = Self::resized(&raw, (width, height), (side, side))?;
                self.push_window(&mut pixels, &thumbnail, side, (0, 0, side, side));
            }
            tiles_per_image.push(tiles as i64);
            feature_token_counts.push(tiles * self.geometry.tile_tokens());
            item_sizes.push((image.width(), image.height()));
        }
        Ok(PreprocessedEncoderInputs::new(
            Self::encoder_input(pixels, [total_tiles, 3, side, side])?,
            feature_token_counts,
            item_sizes,
        )
        .with_extra(
            "image_num_patches",
            ModelSpecificValue::int_1d(tiles_per_image),
        ))
    }

    /// One batch per image, sized together under the call's budget.
    fn preprocess_dynamic(
        &self,
        images: &[&DynamicImage],
        context: &VisionPreprocessingContext,
    ) -> Result<Vec<PreprocessedEncoderInputs>, TransformError> {
        let sizes: Vec<_> = images
            .iter()
            .map(|image| (image.width(), image.height()))
            .collect();
        let p = self.geometry.patch_size;
        let shuffle = self.geometry.shuffle();
        self.plan_grids(&sizes, context.token_budget)?
            .into_iter()
            .zip(images)
            .map(|((rows, cols), image)| {
                let (width, height, raw) = NemotronOmniProcessor::rgb_on_white(image);
                let (out_w, out_h) = (cols * p, rows * p);
                let resized = Self::resized(&raw, (width, height), (out_w, out_h))?;
                let mut pixels = Vec::with_capacity(3 * out_w * out_h);
                self.push_window(&mut pixels, &resized, out_w, (0, 0, out_w, out_h));
                let tokens = rows * cols / (shuffle * shuffle);
                Ok(PreprocessedEncoderInputs::new(
                    Self::encoder_input(pixels, [1, 3, out_h, out_w])?,
                    vec![tokens],
                    vec![(image.width(), image.height())],
                )
                .with_extra(
                    "imgs_sizes",
                    ModelSpecificValue::TupleVec(vec![(out_h as u32, out_w as u32)]),
                )
                .with_extra(
                    "num_tokens_per_image",
                    ModelSpecificValue::IntVec(vec![tokens as i64]),
                ))
            })
            .collect()
    }

    /// One clip's frames, resized to the clip's frame size, with the frames'
    /// source indices and duration.
    fn preprocess_frames(
        &self,
        frames: &[RgbFrameRef<'_>],
        sampling: Option<&VideoSamplingInfo>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let first = frames.first().ok_or(TransformError::EmptyBatch)?;
        let sampling = sampling
            .filter(|s| s.source_fps.is_finite() && s.source_fps > 0.0)
            .ok_or_else(|| {
                TransformError::ShapeError(
                    "Nemotron video needs the frame times the decoder reports (no frame rate)"
                        .to_string(),
                )
            })?;
        if sampling.frame_indices.len() != frames.len() {
            return Err(TransformError::ShapeError(format!(
                "{} frames but {} frame indices",
                frames.len(),
                sampling.frame_indices.len()
            )));
        }
        if first.width == 0 || first.height == 0 {
            return Err(TransformError::ShapeError(format!(
                "a {}x{} video frame has no pixels",
                first.width, first.height
            )));
        }
        let (out_h, out_w) = self.plan_video_size(first.width, first.height);
        let mut pixels = Vec::with_capacity(frames.len() * 3 * out_w * out_h);
        for frame in frames {
            if (frame.width, frame.height) != (first.width, first.height) {
                return Err(TransformError::ShapeError(format!(
                    "Nemotron video frames must share one size, got {}x{} and {}x{}",
                    first.width, first.height, frame.width, frame.height
                )));
            }
            let size = (frame.width as usize, frame.height as usize);
            let resized = Self::resized(frame.data, size, (out_w, out_h))?;
            self.push_window(&mut pixels, &resized, out_w, (0, 0, out_w, out_h));
        }
        let p = self.geometry.patch_size;
        let tokens =
            self.geometry.tubelets(frames.len()) * self.geometry.frame_tokens(out_h / p, out_w / p);
        let indices: Vec<i64> = sampling.frame_indices.iter().map(|&i| i as i64).collect();
        Ok(PreprocessedEncoderInputs::new(
            Self::encoder_input(pixels, [frames.len(), 3, out_h, out_w])?,
            vec![tokens],
            vec![(first.width, first.height)],
        )
        .with_extra(
            "video_num_patches",
            ModelSpecificValue::int_1d(vec![frames.len() as i64]),
        )
        .with_extra(
            "frames_indices",
            ModelSpecificValue::int_2d(indices, 1, frames.len()),
        )
        .with_extra(
            "frame_duration_ms",
            ModelSpecificValue::int_1d(vec![(1000.0 / sampling.source_fps).trunc() as i64]),
        ))
    }
}

impl VisionPreProcessor for NemotronVLProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.geometry.norm_mean
    }

    fn default_std(&self) -> [f64; 3] {
        self.geometry.norm_std
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_with_context(images, &VisionPreprocessingContext::default())
    }

    /// One batch: tiled images always join; dynamic-resolution images only
    /// when they share a size (see
    /// [`preprocess_items_with_context`](Self::preprocess_items_with_context)).
    fn preprocess_with_context(
        &self,
        images: &[&DynamicImage],
        context: &VisionPreprocessingContext,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let mut batches = self.preprocess_items_with_context(images, context)?;
        if batches.len() == 1 {
            return Ok(batches.remove(0));
        }
        let layouts = HashMap::from(
            DYNAMIC_IMAGE_SIDE_TENSORS.map(|key| (key.to_string(), FieldLayout::Batched)),
        );
        PreprocessedEncoderInputs::concat(batches, &layouts)
            .map_err(|e| TransformError::ShapeError(format!("{e:#}")))
    }

    fn preprocess_items_with_context(
        &self,
        images: &[&DynamicImage],
        context: &VisionPreprocessingContext,
    ) -> Result<Vec<PreprocessedEncoderInputs>, TransformError> {
        TransformError::if_empty(images)?;
        if self.geometry.dynamic.is_some() {
            self.preprocess_dynamic(images, context)
        } else {
            Ok(vec![self.preprocess_tiled(images)?])
        }
    }

    /// Refused: the outputs need the frames' source indices
    /// ([`preprocess_video_rgb_sampled`](Self::preprocess_video_rgb_sampled)).
    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        _clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_frames(frames, None)
    }

    fn preprocess_video_rgb_sampled(
        &self,
        frames: &[RgbFrameRef<'_>],
        _clip: &VideoClipOptions,
        sampling: Option<&VideoSamplingInfo>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_frames(frames, sampling)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        if self.geometry.dynamic.is_none() {
            let (columns, rows) = self.plan_tiles(width, height);
            return Ok(self.tile_count(columns, rows) * self.geometry.tile_tokens());
        }
        let (rows, cols) = self.plan_grid(width, height).ok_or_else(|| {
            TransformError::ShapeError(format!(
                "a {width}x{height} image is too thin for Nemotron's patch grid"
            ))
        })?;
        Ok(rows * cols / self.geometry.shuffle().pow(2))
    }

    fn model_name(&self) -> &'static str {
        "nemotron_vl"
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    /// The fields of the three checkpoints' configs the processor reads.
    fn config(kind: &str) -> Value {
        let mut config = json!({
            "patch_size": 16, "downsample_ratio": 0.5, "force_image_size": 512,
            "use_thumbnail": true, "norm_mean": CLIP_MEAN, "norm_std": CLIP_STD,
            "vision_config": {"args": {}}
        });
        match kind {
            // Nemotron Nano 12B v2 VL: tiles, square frames one at a time.
            "v2" => {}
            // Nemotron 3 Nano Omni: dynamic resolution, 1024-patch frames.
            "omni" => {
                config["vision_config"] = json!({
                    "args": {"min_num_patches": 1024, "max_num_patches": 13312},
                    "video_target_num_patches": 1024,
                    "video_temporal_patch_size": 2,
                    "video_maintain_aspect_ratio": true
                });
            }
            // NemotronH_Super_Omni_Reasoning_V3: dynamic resolution, square frames in pairs.
            "super" => {
                config["use_thumbnail"] = json!(false);
                config["vision_config"] = json!({
                    "args": {"min_num_patches": 1024, "max_num_patches": 13312},
                    "video_temporal_patch_size": 2,
                    "video_target_num_patches": null
                });
            }
            other => panic!("{other}"),
        }
        config
    }

    fn processor(kind: &str) -> NemotronVLProcessor {
        NemotronVLProcessor::new(NemotronVLGeometry::from_model_config(&config(kind)).unwrap())
    }

    /// The integers of a model-specific value, as tensor data or a list.
    fn ints(value: Option<&ModelSpecificValue>) -> Vec<i64> {
        match value {
            Some(ModelSpecificValue::IntTensor { data, .. } | ModelSpecificValue::IntVec(data)) => {
                data.clone()
            }
            Some(ModelSpecificValue::TupleVec(pairs)) => pairs
                .iter()
                .flat_map(|&(a, b)| [i64::from(a), i64::from(b)])
                .collect(),
            other => panic!("integers expected, got {other:?}"),
        }
    }

    #[test]
    fn reads_each_checkpoints_geometry() {
        let v2 = processor("v2").geometry;
        assert_eq!(v2.dynamic, None);
        assert_eq!(v2.video_frame_size, VideoFrameSize::Square);
        assert_eq!((v2.video_temporal_patch_size, v2.tile_tokens()), (1, 256));
        let omni = processor("omni").geometry;
        assert_eq!(omni.dynamic, Some((1024, 13312)));
        assert_eq!(
            omni.video_frame_size,
            VideoFrameSize::Patches {
                patches: 1024,
                keep_aspect: true
            }
        );
        let sup = processor("super").geometry;
        assert_eq!(
            (sup.video_frame_size, sup.video_temporal_patch_size),
            (VideoFrameSize::Square, 2)
        );

        let mut sized = config("omni");
        sized["vision_config"]["video_target_num_patches"] = Value::Null;
        sized["vision_config"]["video_target_img_size"] = json!(500);
        assert_eq!(
            NemotronVLGeometry::from_model_config(&sized)
                .unwrap()
                .video_frame_size,
            VideoFrameSize::Patches {
                patches: 32 * 32,
                keep_aspect: true
            }
        );
        sized["vision_config"]["video_target_num_patches"] = json!(1024);
        assert_eq!(
            NemotronVLGeometry::from_model_config(&sized),
            Err(NemotronVLConfigError::Invalid(
                "/vision_config/video_target_img_size"
            ))
        );
        let mut broken = config("v2");
        broken.as_object_mut().unwrap().remove("norm_std");
        assert_eq!(
            NemotronVLGeometry::from_model_config(&broken),
            Err(NemotronVLConfigError::Missing("/norm_std"))
        );
    }

    /// Token counts the engine's tiler gives these sizes at the per-image maximum.
    #[test]
    fn plans_the_reference_grids() {
        let processor = processor("omni");
        let tokens = |w, h| {
            let (rows, cols) = processor.plan_grid(w, h).unwrap();
            rows * cols / 4
        };
        assert_eq!(tokens(224, 224), 256);
        assert_eq!(tokens(640, 480), 300);
        assert_eq!(tokens(1280, 720), 920);
        assert_eq!(tokens(1920, 1080), 2040);
        assert_eq!(tokens(2048, 2048), 3306);
        assert_eq!(tokens(4000, 3000), 3300);
        // 528 px is 33 patches plus a half, which Python rounds to even: 34.
        assert_eq!(processor.plan_grid(528, 528), Some((34, 34)));
        // Rounds to a zero-height grid; the engine divides by its area and fails.
        assert_eq!(processor.plan_grid(300_000, 8), None);
        let error = processor
            .calculate_num_tokens(300_000, 8)
            .unwrap_err()
            .to_string();
        assert!(error.contains("too thin"), "{error}");
    }

    /// Grids the engine's `compute_params` gives these images when
    /// `max_model_len - prompt_len` is the token budget.
    #[test]
    fn plans_the_reference_grids_under_a_token_budget() {
        let processor = processor("omni");
        let plan = |budget, sizes: &[(u32, u32)]| processor.plan_grids(sizes, budget).unwrap();
        let pair = [(1920, 1080), (640, 480)];
        assert_eq!(plan(None, &pair), [(68, 120), (30, 40)]);
        assert_eq!(plan(Some(3004), &pair), [(68, 120), (30, 40)]);
        assert_eq!(plan(Some(1504), &[(1920, 1080)]), [(58, 102)]);
        // The tiler keeps 4 tokens back: 304 tokens leave 300 for a 30x40 grid,
        // 303 do not.
        assert_eq!(plan(Some(304), &[(640, 480)]), [(30, 40)]);
        assert_eq!(plan(Some(303), &[(640, 480)]), [(30, 38)]);
        assert_eq!(
            plan(Some(3004), &[(1920, 1080), (1920, 1080)]),
            [(58, 102), (58, 102)]
        );
        // A small image whose share falls to the minimum keeps its own grid.
        assert_eq!(
            plan(Some(4004), &[(4000, 3000), (224, 224), (1280, 720)]),
            [(94, 124), (14, 14), (42, 76)]
        );
        assert_eq!(plan(Some(0), &[(640, 480)]), [(28, 36)]);
        assert_eq!(plan(Some(100), &pair), [(24, 42), (28, 36)]);
    }

    /// Dynamic images: one `(1, 3, H, W)` batch per image with its grid in
    /// pixels and its token count, sized together under the budget.
    #[test]
    fn sizes_dynamic_images_one_batch_each() {
        let processor = processor("omni");
        let images = [
            &DynamicImage::new_rgb8(640, 480),
            &DynamicImage::new_rgb8(1920, 1080),
        ];
        let context = |token_budget| VisionPreprocessingContext { token_budget };
        let batches = processor
            .preprocess_items_with_context(&images, &context(None))
            .unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].encoder_input.shape(), &[1, 3, 480, 640]);
        assert_eq!(batches[1].encoder_input.shape(), &[1, 3, 1088, 1920]);
        assert_eq!(batches[1].feature_token_counts, [2040]);
        assert_eq!(
            ints(batches[1].model_specific.get("imgs_sizes")),
            vec![1088, 1920]
        );
        assert_eq!(
            ints(batches[1].model_specific.get("num_tokens_per_image")),
            vec![2040]
        );
        let budgeted = processor
            .preprocess_items_with_context(&images[..1], &context(Some(303)))
            .unwrap();
        assert_eq!(budgeted[0].encoder_input.shape(), &[1, 3, 480, 608]);
        // Two images of one size still join into one batch.
        let joined = processor
            .preprocess_with_context(&[images[0], images[0]], &context(None))
            .unwrap();
        assert_eq!(joined.encoder_input.shape(), &[2, 3, 480, 640]);
    }

    /// Tile grids of `find_closest_aspect_ratio` at the default 12 tiles.
    #[test]
    fn plans_the_reference_tiles() {
        let processor = processor("v2");
        let tiles = |w, h| processor.plan_tiles(w, h);
        assert_eq!(tiles(1920, 1080), (4, 2));
        assert_eq!(tiles(640, 480), (4, 3));
        assert_eq!(tiles(512, 512), (1, 1));
        assert_eq!(tiles(300, 1200), (1, 4));
        // Equally close to 1:1 at every size: the larger grid once the image
        // covers more than half of it.
        assert_eq!(tiles(1024, 1024), (2, 2));
        assert_eq!(processor.calculate_num_tokens(1920, 1080).unwrap(), 9 * 256);
    }

    /// Tiles row by row, then the thumbnail, as `(tiles, 3, 512, 512)`.
    #[test]
    fn cuts_tiles_row_by_row_and_appends_the_thumbnail() {
        let processor = processor("v2");
        // 1024x512 is a 2x1 grid: left half dark, right half bright.
        let mut image = image::RgbImage::new(1024, 512);
        for (x, _, px) in image.enumerate_pixels_mut() {
            *px = image::Rgb([if x < 512 { 0 } else { 255 }; 3]);
        }
        let image = DynamicImage::from(image);
        let out = processor
            .preprocess(&[&image, &DynamicImage::new_rgb8(512, 512)])
            .unwrap();
        // 2 tiles and a thumbnail, then a single tile without one.
        assert_eq!(out.encoder_input.shape(), &[4, 3, 512, 512]);
        assert_eq!(out.feature_token_counts, [768, 256]);
        assert_eq!(
            ints(out.model_specific.get("image_num_patches")),
            vec![3, 1]
        );
        let values = out.encoder_input.as_f32();
        let normalized = |v: f32| (v / 255.0 - CLIP_MEAN[0] as f32) / CLIP_STD[0] as f32;
        let (black, white) = (normalized(0.0), normalized(255.0));
        assert_eq!((values[[0, 0, 0, 0]], values[[1, 0, 0, 0]]), (black, white));
        // The thumbnail squeezes both halves into one tile.
        assert_eq!(values[[2, 0, 0, 0]], black);
        assert!((values[[2, 0, 0, 511]] - white).abs() < 1e-4);
    }

    /// Frame sizes `get_video_target_size_and_feature_size` gives these clips.
    #[test]
    fn plans_the_reference_video_sizes() {
        let omni = processor("omni");
        assert_eq!(omni.plan_video_size(1920, 1080), (384, 672));
        assert_eq!(omni.plan_video_size(640, 480), (448, 576));
        assert_eq!(omni.plan_video_size(512, 512), (512, 512));
        // Square roots within 1e-9 of a half, which Python rounds to even.
        assert_eq!(omni.plan_video_size(4096, 841), (14 * 16, 70 * 16));
        assert_eq!(omni.plan_video_size(16, 159), (102 * 16, 10 * 16));
        assert_eq!(processor("super").plan_video_size(1920, 1080), (512, 512));
        assert_eq!(processor("v2").plan_video_size(1920, 1080), (512, 512));
    }

    /// A clip's frames in order, with their source indices, duration and
    /// tubelet tokens.
    #[test]
    fn sends_frames_with_their_times() {
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
        let clip = VideoClipOptions::sampled_at(30.0);
        let sampling = VideoSamplingInfo {
            source_fps: 30.0,
            frame_indices: vec![0, 15, 29],
        };
        let omni = processor("omni")
            .preprocess_video_rgb_sampled(&refs, &clip, Some(&sampling))
            .unwrap();
        // A 32x32 frame is a 32x32-patch grid; two tubelets (the third frame
        // is padded by the engine) of 256 tokens.
        assert_eq!(omni.encoder_input.shape(), &[3, 3, 512, 512]);
        assert_eq!(omni.feature_token_counts, [2 * 256]);
        assert_eq!(
            ints(omni.model_specific.get("frames_indices")),
            vec![0, 15, 29]
        );
        assert_eq!(ints(omni.model_specific.get("frame_duration_ms")), vec![33]);
        let values = omni.encoder_input.as_f32();
        let first = |frame: usize| values[[frame, 0, 0, 0]];
        assert!(first(0) < first(1) && first(1) < first(2));
        // One frame per tubelet on V2.
        assert_eq!(
            processor("v2")
                .preprocess_video_rgb_sampled(&refs, &clip, Some(&sampling))
                .unwrap()
                .feature_token_counts,
            [3 * 256]
        );
        // No frame rate, no timestamps.
        assert!(processor("omni")
            .preprocess_video_rgb(&refs, &clip)
            .is_err());
        // A frame already at the target size still has its buffer checked.
        let short = vec![0u8; 512 * 512 * 3 - 1];
        let frame = [RgbFrameRef {
            width: 512,
            height: 512,
            data: &short,
        }];
        let one = VideoSamplingInfo {
            source_fps: 30.0,
            frame_indices: vec![0],
        };
        assert!(processor("omni")
            .preprocess_video_rgb_sampled(&frame, &clip, Some(&one))
            .is_err());
    }
}
