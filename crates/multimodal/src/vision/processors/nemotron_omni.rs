//! Nemotron VL / Omni (C-RADIO) image and video preprocessing.
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
//! The engine's tiler scales a request's images down to share `4 *
//! (max_model_len - text_prompt_length - 4)` patches. This port plans the same
//! grids when the caller passes `max_model_len - text_prompt_length` as the
//! call's [`token_budget`](VisionPreprocessingContext::token_budget); which
//! text length to subtract is the caller's choice. Without a budget, each
//! image gets the per-image maximum. A budget too small for every image's
//! minimum still gives each its minimum, as the engine does, and the
//! request's length check refuses the prompt.
//!
//! That is [`NemotronOmniProcessor::new`]: Nemotron 3 Nano Omni in the packed
//! layout. The engine's processor (`NanoNemotronVLProcessor`) serves the
//! family's other checkpoints the same way from their model config, read
//! here as [`NemotronGeometry`] together with what the processor config names
//! in its place
//! ([`with_preprocessor_config`](NemotronGeometry::with_preprocessor_config));
//! the patch size, patch counts, shuffle factor and normalization then come
//! from the checkpoint, and
//!
//! - a config without `min_num_patches` tiles its images, as InternVL's
//!   `dynamic_preprocess` does: the image is resized to the tile grid whose
//!   aspect is closest to its own (at most 12 tiles of `force_image_size`
//!   pixels), cut into tiles row by row, and followed by a thumbnail of the
//!   whole image when `use_thumbnail` and there is more than one tile, `S * S
//!   / p^2 * r^2` tokens per tile;
//! - video frames are resized to about `video_target_num_patches` patches (or
//!   `ceil(video_target_img_size / p)^2`), keeping the clip's aspect when
//!   `video_maintain_aspect_ratio`, else to `force_image_size` squares, in
//!   tubelets of `video_temporal_patch_size`.
//!
//! [`NemotronOmniProcessor::reference`] serves any of these geometries in the
//! engine's own tensors: `pixel_values_flat` as dynamic-resolution
//! `(patches, 3 * p * p)` rows or tiled `(tiles, 3, S, S)` pixels with
//! `image_num_patches`, dynamic-resolution images additionally carrying
//! `imgs_sizes` and `num_tokens_per_image`, and `pixel_values_flat_video` as
//! `(frames, 3, H, W)` with `video_num_patches`, the sampled `frames_indices`
//! and `frame_duration_ms`.

use std::borrow::Cow;

use image::DynamicImage;
use ndarray::{Array2, ArrayD, IxDyn};
use serde_json::{json, Value};

use crate::{
    encoder_inputs::ModelSpecificValue,
    types::{RgbFrameRef, VideoSamplingInfo},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{
            PreprocessedEncoderInputs, VideoClipOptions, VisionPreProcessor,
            VisionPreprocessingContext,
        },
        transforms::{
            resize_bicubic_torch_aa_f32, resize_bicubic_torch_aa_f32_strips, rgb_bytes,
            TransformError,
        },
    },
};

/// Nemotron 3 Nano Omni's geometry, the one the packed layout serves:
/// [`NemotronOmniProcessor::new`] plans with it, and
/// [`mismatched_config_field`](NemotronOmniProcessor::mismatched_config_field)
/// holds a checkpoint to it.
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
/// Tokens the engine keeps back from a request's image budget
/// (`max_num_tokens_available`).
const BUDGET_RESERVED_TOKENS: usize = 4;
/// Rounds the engine's `compute_params` takes to fit a request's images in
/// their budget before it gives up.
const MAX_BUDGET_ROUNDS: usize = 10;

/// Embeddings of a `rows x cols` patch grid after the 2x2 pixel shuffle.
pub(crate) fn grid_tokens(rows: usize, cols: usize) -> usize {
    rows * cols / (GRID_DIVISOR * GRID_DIVISOR)
}
const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];
/// The engine's tile cap (`DEFAULT_NUM_TILES`).
const DEFAULT_MAX_NUM_TILES: usize = 12;
/// A dynamic-resolution image's reference tensors beside its pixels, one
/// entry per image.
pub(crate) const DYNAMIC_IMAGE_SIDE_TENSORS: [&str; 3] =
    ["imgs_sizes", "num_tokens_per_image", "image_num_patches"];

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

/// A geometry field a config lacks, or holds with a value the engine would
/// refuse; each names the field, as a JSON pointer into the model config or
/// as a key of the processor config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronConfigError {
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
pub struct NemotronGeometry {
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
    /// The most tiles of one image (`max_num_tiles`), its thumbnail apart.
    pub max_num_tiles: usize,
}

impl NemotronGeometry {
    /// Read the geometry from a model config.
    pub fn from_model_config(config: &Value) -> Result<Self, NemotronConfigError> {
        use NemotronConfigError::{Invalid, Missing};
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
                let max = match number("/vision_config/args/max_num_patches")? {
                    max if max <= 0.0 => usize::MAX,
                    max if max.fract() == 0.0 => max as usize,
                    _ => return Err(Invalid("/vision_config/args/max_num_patches")),
                };
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
            // The writer divides by the std as `f32`.
            norm_std: triple("/norm_std").and_then(|std| {
                let usable = |&v: &f64| (v as f32).is_finite() && v as f32 > 0.0;
                std.iter()
                    .all(usable)
                    .then_some(std)
                    .ok_or(Invalid("/norm_std"))
            })?,
            dynamic,
            video_frame_size,
            video_temporal_patch_size: optional_count("/vision_config/video_temporal_patch_size")?
                .unwrap_or(1),
            max_num_tiles: DEFAULT_MAX_NUM_TILES,
        })
    }

    /// The geometry with the values the checkpoint's processor config
    /// (`preprocessor_config.json`) names in place of the model config's: the
    /// video frame size (`video_target_num_patches`,
    /// `video_maintain_aspect_ratio`) and the tile cap (`max_num_tiles`). A
    /// `null` names nothing. Nemotron 3.5 Super VL sizes its video frames only
    /// there, so its frames follow the checkpoint's own processor rather than
    /// the square frames the engine's processor reads from the model config.
    pub fn with_preprocessor_config(
        mut self,
        config: &PreProcessorConfig,
    ) -> Result<Self, NemotronConfigError> {
        use NemotronConfigError::Invalid;
        let named = |key: &'static str| config.extra.get(key).filter(|v| !v.is_null());
        let count = |key: &'static str| match (named(key), config.extra_usize(key)) {
            (None, _) => Ok(None),
            (Some(_), Ok(Some(count))) if count >= 1 => Ok(Some(count)),
            _ => Err(Invalid(key)),
        };
        let keep_aspect = named("video_maintain_aspect_ratio")
            .map(|flag| flag.as_bool().ok_or(Invalid("video_maintain_aspect_ratio")))
            .transpose()?;
        let (own_patches, own_keep_aspect) = match self.video_frame_size {
            VideoFrameSize::Patches {
                patches,
                keep_aspect,
            } => (Some(patches), keep_aspect),
            VideoFrameSize::Square => (None, false),
        };
        if let Some(patches) = count("video_target_num_patches")?.or(own_patches) {
            self.video_frame_size = VideoFrameSize::Patches {
                patches,
                keep_aspect: keep_aspect.unwrap_or(own_keep_aspect),
            };
        }
        if let Some(max_num_tiles) = count("max_num_tiles")? {
            self.max_num_tiles = max_num_tiles;
        }
        Ok(self)
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

/// The engine's tile grid candidates `(columns, rows)` for up to `max_tiles`
/// tiles, by area (`get_internvl_target_ratios`).
fn target_ratios(max_tiles: usize) -> Vec<(usize, usize)> {
    let mut ratios: Vec<_> = (1..=max_tiles)
        .flat_map(|columns| (1..=max_tiles / columns).map(move |rows| (columns, rows)))
        .collect();
    ratios.sort_by_key(|(columns, rows)| columns * rows);
    ratios
}

/// The tensors the pixels leave in (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Packed,
    Reference,
}

/// C-RADIO image and video preprocessor for the Nemotron VL / Omni family.
#[derive(Debug, Clone)]
pub struct NemotronOmniProcessor {
    geometry: NemotronGeometry,
    layout: Layout,
}

impl Default for NemotronOmniProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl NemotronOmniProcessor {
    /// The Packed processor of Nemotron 3 Nano Omni. Its write path is tuned
    /// for that checkpoint's 16-pixel patch, so the geometry is not a
    /// parameter.
    pub fn new() -> Self {
        Self {
            geometry: NemotronGeometry {
                patch_size: PATCH_SIZE,
                downsample_ratio: 0.5,
                image_size: 512,
                use_thumbnail: true,
                norm_mean: CLIP_MEAN,
                norm_std: CLIP_STD,
                dynamic: Some((MIN_NUM_PATCHES, MAX_NUM_PATCHES)),
                video_frame_size: VideoFrameSize::Patches {
                    patches: VIDEO_TARGET_PATCHES,
                    keep_aspect: true,
                },
                video_temporal_patch_size: VIDEO_TEMPORAL_PATCH_SIZE,
                max_num_tiles: DEFAULT_MAX_NUM_TILES,
            },
            layout: Layout::Packed,
        }
    }

    /// The Reference processor of a checkpoint's geometry.
    pub fn reference(geometry: NemotronGeometry) -> Self {
        Self {
            geometry,
            layout: Layout::Reference,
        }
    }

    /// The checkpoint config field, as a JSON pointer, whose value differs from
    /// the geometry this port implements, if any.
    pub(crate) fn mismatched_config_field(config: &Value) -> Option<&'static str> {
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
    /// thin it rounds to no patches, which the engine fails on too, or for a
    /// tiling geometry.
    pub(crate) fn plan_grid(&self, width: u32, height: u32) -> Option<(usize, usize)> {
        let (_, max) = self.geometry.dynamic?;
        self.plan_grid_within(width, height, max)
    }

    /// [`plan_grid`](Self::plan_grid) within `budget` patches, the engine's
    /// `num_tokens_available` for the image. The minimum is enforced only
    /// above it: an image whose budget is the minimum keeps the grid the
    /// budget scales it to, however small.
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
                    "a {width}x{height} image is too thin for Nemotron-H Omni's patch grid"
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
            "Nemotron-H Omni's images did not fit a {token_budget}-token budget in \
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
        for (columns, rows) in target_ratios(self.geometry.max_num_tiles) {
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
    /// (`get_video_target_size_and_feature_size`, with the engine's
    /// `_compute_aspect_preserving_size`, else `force_image_size`).
    pub fn plan_video_size(&self, width: u32, height: u32) -> (usize, usize) {
        let p = self.geometry.patch_size;
        let divisor = self.geometry.shuffle();
        match self.geometry.video_frame_size {
            VideoFrameSize::Square => (self.geometry.image_size, self.geometry.image_size),
            VideoFrameSize::Patches {
                patches,
                keep_aspect: false,
            } => {
                let side = divisor.max((patches as f64).sqrt() as usize / divisor * divisor);
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

    /// The image's RGB bytes with any transparency composited onto white the
    /// way the engine's loader does it: PIL's `paste` with the alpha as mask.
    fn rgb_on_white(image: &DynamicImage) -> (usize, usize, Cow<'_, [u8]>) {
        if !image.color().has_alpha() {
            return rgb_bytes(image);
        }
        // An RGBA8 input is read where it stands; another layout with alpha
        // is converted first.
        let rgba = image
            .as_rgba8()
            .map_or_else(|| Cow::Owned(image.to_rgba8()), Cow::Borrowed);
        let (width, height) = (rgba.width() as usize, rgba.height() as usize);
        // The container may be longer than its pixels.
        let (pixels, _) = rgba.as_raw()[..4 * width * height].as_chunks::<4>();
        // Whole pixels are collected and the vector is flattened: one exact
        // allocation, and no capacity check per byte in the loop.
        let out = pixels
            .iter()
            .map(|px| {
                let alpha = u32::from(px[3]);
                // PIL's BLEND: (255 * (255 - a) + c * a) / 255, rounded with its DIV255.
                [px[0], px[1], px[2]].map(|c| {
                    let v = 255 * (255 - alpha) + u32::from(c) * alpha + 128;
                    ((v + (v >> 8)) >> 8) as u8
                })
            })
            .collect::<Vec<[u8; 3]>>()
            .into_flattened();
        (width, height, Cow::Owned(out))
    }

    /// Resize, normalize and patchify one RGB image: appends its
    /// `rows * cols * 3 * p * p` values to `out`.
    fn patches(
        raw: &[u8],
        width: u32,
        height: u32,
        rows: usize,
        cols: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), TransformError> {
        let p = PATCH_SIZE;
        let (out_w, out_h) = (cols * p, rows * p);
        let scale = CLIP_STD.map(|s| s as f32);
        let mean = CLIP_MEAN.map(|m| m as f32);
        // The resize hands over one row of patches at a time, so the resized
        // image never stands whole in memory.
        let patch_row = move |strip: &[f32]| {
            let (pixels, _) = strip.as_chunks::<3>();
            for j in 0..cols {
                for c in 0..3 {
                    let (mean, scale) = (mean[c], scale[c]);
                    for dy in 0..PATCH_SIZE {
                        let row = &pixels[dy * out_w + j * PATCH_SIZE..][..PATCH_SIZE];
                        // One channel of a patch row is gathered into a fixed array,
                        // then normalized there with that channel's mean and scale:
                        // the resized image is read once, and the sixteen divisions
                        // run as one vector block. Normalizing inside the gather
                        // divides one value at a time and is slower.
                        let mut values = [0.0_f32; PATCH_SIZE];
                        for (value, pixel) in values.iter_mut().zip(row) {
                            *value = pixel[c];
                        }
                        // f32 op order matches the reference: x / 255, - mean, / std.
                        for value in &mut values {
                            *value = (*value / 255.0 - mean) / scale;
                        }
                        out.extend_from_slice(&values);
                    }
                }
            }
        };
        let (w, h) = (out_w as u32, out_h as u32);
        resize_bicubic_torch_aa_f32_strips(raw, width, height, w, h, p, patch_row)?;
        Ok(())
    }

    /// Packed: one clip's frames as patches plus its `video_grid_thw`.
    #[expect(
        clippy::expect_used,
        reason = "the patch buffer is sized from the grid it is reshaped by"
    )]
    fn video_patches(
        &self,
        frames: &[RgbFrameRef<'_>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let first = frames.first().ok_or(TransformError::EmptyBatch)?;
        let (out_h, out_w) = self.plan_video_size(first.width, first.height);
        let (rows, cols) = (out_h / PATCH_SIZE, out_w / PATCH_SIZE);
        let patch_len = 3 * PATCH_SIZE * PATCH_SIZE;
        let mut encoder_input = Vec::with_capacity(frames.len() * rows * cols * patch_len);
        for frame in frames {
            if (frame.width, frame.height) != (first.width, first.height) {
                return Err(TransformError::ShapeError(format!(
                    "Nemotron-H Omni video frames must share one size, got {}x{} and {}x{}",
                    first.width, first.height, frame.width, frame.height
                )));
            }
            Self::patches(
                frame.data,
                frame.width,
                frame.height,
                rows,
                cols,
                &mut encoder_input,
            )?;
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

    /// Reference: append the `(x, y, w, h)` window of a resized image
    /// `stride` pixels wide, channel-major, as `(x / 255 - mean) / std`.
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

    /// Reference: one batch of a call's tiled images.
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
            let (width, height, raw) = Self::rgb_on_white(image);
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

    /// Reference: one patch-flat batch of a call's dynamic-resolution images,
    /// sized together under the call's budget.
    fn preprocess_dynamic(
        &self,
        images: &[&DynamicImage],
        context: &VisionPreprocessingContext,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let sizes: Vec<_> = images
            .iter()
            .map(|image| (image.width(), image.height()))
            .collect();
        let p = self.geometry.patch_size;
        let shuffle = self.geometry.shuffle();
        let planned = self.plan_grids(&sizes, context.token_budget)?;
        let total_patches: usize = planned.iter().map(|&(rows, cols)| rows * cols).sum();
        let mut pixels = Vec::with_capacity(total_patches * 3 * p * p);
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut resized_sizes = Vec::with_capacity(images.len());
        let mut tokens_per_image = Vec::with_capacity(images.len());
        let mut patches_per_image = Vec::with_capacity(images.len());
        for (image, &(rows, cols)) in images.iter().zip(&planned) {
            let (width, height, raw) = Self::rgb_on_white(image);
            let (out_w, out_h) = (cols * p, rows * p);
            let resized = Self::resized(&raw, (width, height), (out_w, out_h))?;
            for row in 0..rows {
                for column in 0..cols {
                    self.push_window(&mut pixels, &resized, out_w, (column * p, row * p, p, p));
                }
            }
            let tokens = rows * cols / (shuffle * shuffle);
            feature_token_counts.push(tokens);
            item_sizes.push((image.width(), image.height()));
            resized_sizes.push((out_h as u32, out_w as u32));
            tokens_per_image.push(tokens as i64);
            patches_per_image.push((rows * cols) as i64);
        }
        let encoder_input = Array2::from_shape_vec((total_patches, 3 * p * p), pixels)
            .map_err(|e| TransformError::ShapeError(format!("dynamic image patches: {e}")))?;
        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra("imgs_sizes", ModelSpecificValue::TupleVec(resized_sizes))
                .with_extra(
                    "num_tokens_per_image",
                    ModelSpecificValue::IntVec(tokens_per_image),
                )
                .with_extra(
                    "image_num_patches",
                    ModelSpecificValue::int_1d(patches_per_image),
                ),
        )
    }

    /// Reference: one clip's frames, resized to the clip's frame size, with
    /// the frames' source indices and duration.
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

impl VisionPreProcessor for NemotronOmniProcessor {
    /// The packed layout only, and only without a token budget, which the
    /// call's images share.
    fn supports_per_image_preprocessing(&self) -> bool {
        self.layout == Layout::Packed
    }

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

    /// One batch: packed and tiled images always join, and reference
    /// dynamic-resolution images join as flat patches whatever their sizes.
    #[expect(
        clippy::expect_used,
        reason = "the patch buffer is filled from the planned grids it is reshaped by"
    )]
    fn preprocess_with_context(
        &self,
        images: &[&DynamicImage],
        context: &VisionPreprocessingContext,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        TransformError::if_empty(images)?;
        if self.layout == Layout::Reference {
            return match self.geometry.dynamic {
                Some(_) => self.preprocess_dynamic(images, context),
                None => self.preprocess_tiled(images),
            };
        }
        let sizes: Vec<_> = images
            .iter()
            .map(|image| (image.width(), image.height()))
            .collect();
        let planned = self.plan_grids(&sizes, context.token_budget)?;
        let patch_len = 3 * PATCH_SIZE * PATCH_SIZE;
        let mut encoder_input = Vec::with_capacity(
            planned
                .iter()
                .map(|(rows, cols)| rows * cols * patch_len)
                .sum(),
        );
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len() * 2);
        let mut patches_per_image = Vec::with_capacity(images.len());
        for (image, &(rows, cols)) in images.iter().zip(&planned) {
            let (width, height, raw) = Self::rgb_on_white(image);
            Self::patches(
                &raw,
                width as u32,
                height as u32,
                rows,
                cols,
                &mut encoder_input,
            )?;
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
        clip: &VideoClipOptions,
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
        self.preprocess_video_rgb(&refs, clip)
    }

    /// Refused in the reference layout: its outputs need the frames' source
    /// indices
    /// ([`preprocess_video_rgb_sampled`](Self::preprocess_video_rgb_sampled)).
    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_video_rgb_sampled(frames, clip, None)
    }

    fn preprocess_video_rgb_sampled(
        &self,
        frames: &[RgbFrameRef<'_>],
        _clip: &VideoClipOptions,
        sampling: Option<&VideoSamplingInfo>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        match self.layout {
            Layout::Packed => self.video_patches(frames),
            Layout::Reference => self.preprocess_frames(frames, sampling),
        }
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        if self.geometry.dynamic.is_none() {
            let (columns, rows) = self.plan_tiles(width, height);
            return Ok(self.tile_count(columns, rows) * self.geometry.tile_tokens());
        }
        let (rows, cols) = self.plan_grid(width, height).ok_or_else(|| {
            TransformError::ShapeError(format!(
                "a {width}x{height} image is too thin for Nemotron-H Omni's patch grid"
            ))
        })?;
        Ok(rows * cols / self.geometry.shuffle().pow(2))
    }

    fn model_name(&self) -> &'static str {
        match self.layout {
            Layout::Packed => "nemotron_omni",
            Layout::Reference => "nemotron_vl",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The processor config names what the model config leaves out: Nemotron
    /// 3.5 Super VL's video frame size, and the tile cap.
    #[test]
    fn the_processor_config_overrides_the_model_config() {
        let overlay = |kind, processor_config: Value| {
            NemotronGeometry::from_model_config(&config(kind))
                .unwrap()
                .with_preprocessor_config(
                    &PreProcessorConfig::from_value(processor_config).unwrap(),
                )
        };
        let own = overlay("super", json!({})).unwrap();
        assert_eq!(own.video_frame_size, VideoFrameSize::Square);
        let sized = overlay(
            "super",
            json!({"video_target_num_patches": 1024, "video_maintain_aspect_ratio": true}),
        )
        .unwrap();
        assert_eq!(
            NemotronOmniProcessor::reference(sized).plan_video_size(1920, 1080),
            (384, 672)
        );
        let one_tile = overlay("v2", json!({"max_num_tiles": 1})).unwrap();
        assert_eq!(
            NemotronOmniProcessor::reference(one_tile).plan_tiles(1920, 1080),
            (1, 1)
        );
        // A value the processor config does not name, or names `null`, stays
        // the model config's.
        let renamed = overlay(
            "omni",
            json!({"video_target_num_patches": 256, "max_num_tiles": null}),
        )
        .unwrap();
        assert_eq!(
            (renamed.video_frame_size, renamed.max_num_tiles),
            (
                VideoFrameSize::Patches {
                    patches: 256,
                    keep_aspect: true
                },
                12
            )
        );
        let squashed = overlay("omni", json!({"video_maintain_aspect_ratio": false})).unwrap();
        assert_eq!(
            squashed.video_frame_size,
            VideoFrameSize::Patches {
                patches: 1024,
                keep_aspect: false
            }
        );
        for (key, value) in [
            ("max_num_tiles", json!(0)),
            ("video_target_num_patches", json!("1024")),
            ("video_maintain_aspect_ratio", json!(1)),
        ] {
            assert_eq!(
                overlay("omni", json!({ key: value })),
                Err(NemotronConfigError::Invalid(key))
            );
        }
    }

    /// Token counts the engine's tiler gives these sizes at the per-image maximum.
    #[test]
    fn plans_the_reference_grids() {
        let processor = NemotronOmniProcessor::new();
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
        // The count refuses what the grid planner refuses, as the
        // preprocessing does.
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
        let plan = |budget, sizes: &[(u32, u32)]| {
            NemotronOmniProcessor::new()
                .plan_grids(sizes, budget)
                .unwrap()
        };
        let pair = [(1920, 1080), (640, 480)];
        // Without a budget, or with one the images fit, each image gets the
        // per-image maximum.
        assert_eq!(plan(None, &pair), [(68, 120), (30, 40)]);
        assert_eq!(plan(Some(3004), &pair), [(68, 120), (30, 40)]);
        // One image scaled down to fit.
        assert_eq!(plan(Some(1504), &[(1920, 1080)]), [(58, 102)]);
        // The tiler keeps 4 tokens back: 304 tokens leave 300 for a 30x40 grid,
        // 303 do not.
        assert_eq!(plan(Some(304), &[(640, 480)]), [(30, 40)]);
        assert_eq!(plan(Some(303), &[(640, 480)]), [(30, 38)]);
        // Images over a shared budget shrink in proportion to their patches.
        assert_eq!(
            plan(Some(3004), &[(1920, 1080), (1920, 1080)]),
            [(58, 102), (58, 102)]
        );
        // A small image whose share falls to the minimum keeps its own grid,
        // below the minimum, rather than being raised to it.
        assert_eq!(
            plan(Some(4004), &[(4000, 3000), (224, 224), (1280, 720)]),
            [(94, 124), (14, 14), (42, 76)]
        );
        // A budget below the minimum: each image still gets the minimum share.
        assert_eq!(plan(Some(0), &[(640, 480)]), [(28, 36)]);
        assert_eq!(plan(Some(100), &pair), [(24, 42), (28, 36)]);
    }

    /// The budget reaches the engine's inputs: fewer patches and tokens per
    /// image, and a budget below the minimum is served at the minimum rather
    /// than refused.
    #[test]
    fn a_token_budget_shrinks_the_images_it_cannot_fit() {
        let owned = vec![DynamicImage::new_rgb8(640, 480); 2];
        let images: Vec<&DynamicImage> = owned.iter().collect();
        let preprocess = |images: &[&DynamicImage], token_budget| {
            NemotronOmniProcessor::new()
                .preprocess_with_context(images, &VisionPreprocessingContext { token_budget })
                .unwrap()
        };
        let grids = |out: &PreprocessedEncoderInputs| match out.model_specific.get("image_grid_hw")
        {
            Some(ModelSpecificValue::IntTensor { data, .. }) => data.clone(),
            other => panic!("image_grid_hw: {other:?}"),
        };

        let unbudgeted = preprocess(&images, None);
        assert_eq!(unbudgeted.feature_token_counts, [300, 300]);
        assert_eq!(grids(&unbudgeted), [30, 40, 30, 40]);

        let budgeted = preprocess(&images, Some(560));
        assert_eq!(budgeted.feature_token_counts, [266, 266]);
        assert_eq!(grids(&budgeted), [28, 38, 28, 38]);
        assert_eq!(budgeted.encoder_input.shape()[0], 2 * 28 * 38);

        let exhausted = preprocess(&images[..1], Some(0));
        assert_eq!(exhausted.feature_token_counts, [252]);
        assert_eq!(grids(&exhausted), [28, 36]);
    }

    /// Every value of a patch is its resized pixel's, normalized in the
    /// reference's `f32` operation order, at its place: patches row-major,
    /// then channel, row, column.
    #[test]
    fn a_patch_value_is_its_resized_pixel_normalized_at_its_place() {
        let (width, height, rows, cols) = (37_u32, 29_u32, 2_usize, 4_usize);
        let raw: Vec<u8> = (0..width * height * 3)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        let mut patches = Vec::new();
        NemotronOmniProcessor::patches(&raw, width, height, rows, cols, &mut patches).unwrap();

        let p = PATCH_SIZE;
        let resized =
            resize_bicubic_torch_aa_f32(&raw, width, height, (cols * p) as u32, (rows * p) as u32)
                .unwrap();
        let mut expected = Vec::with_capacity(patches.len());
        for i in 0..rows {
            for j in 0..cols {
                for c in 0..3 {
                    for dy in 0..p {
                        for dx in 0..p {
                            let value = resized[((i * p + dy) * cols * p + j * p + dx) * 3 + c];
                            expected
                                .push((value / 255.0 - CLIP_MEAN[c] as f32) / CLIP_STD[c] as f32);
                        }
                    }
                }
            }
        }
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&patches), bits(&expected));
    }

    /// Frame grids the engine's `_compute_aspect_preserving_size` gives these sizes.
    #[test]
    fn plans_the_reference_video_grids() {
        let grid = |w, h| {
            let (height, width) = NemotronOmniProcessor::new().plan_video_size(w, h);
            (height / PATCH_SIZE, width / PATCH_SIZE)
        };
        assert_eq!(grid(1920, 1080), (24, 42));
        assert_eq!(grid(1280, 720), (24, 42));
        assert_eq!(grid(640, 480), (28, 36));
        assert_eq!(grid(512, 512), (32, 32));
        // Square roots within 1e-9 of a half, which Python rounds to even.
        assert_eq!(grid(4096, 841), (14, 70));
        assert_eq!(grid(4096, 3969), (30, 32));
        // An odd row count rounds up to an even grid within the target.
        assert_eq!(grid(16, 159), (102, 10));
        // Without a patch target, `force_image_size` squares.
        assert_eq!(processor("super").plan_video_size(1920, 1080), (512, 512));
        assert_eq!(processor("v2").plan_video_size(1920, 1080), (512, 512));
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

    /// An RGBA8 image in a container longer than its pixels composites as
    /// the exact image does.
    #[test]
    fn a_longer_container_composites_as_its_image() {
        let pixels = vec![0, 0, 0, 0, 200, 10, 99, 128, 37, 250, 1, 1, 9, 8, 7, 255];
        let mut container = pixels.clone();
        container.extend_from_slice(&[0xAB; 7]);
        let exact = DynamicImage::from(image::RgbaImage::from_raw(4, 1, pixels).unwrap());
        let longer = DynamicImage::from(image::RgbaImage::from_raw(4, 1, container).unwrap());
        assert_eq!(
            NemotronOmniProcessor::rgb_on_white(&longer),
            NemotronOmniProcessor::rgb_on_white(&exact)
        );
    }

    /// An input of another layout with alpha is converted to RGBA8 first and
    /// composites as that image does.
    #[test]
    fn another_alpha_layout_composites_as_its_rgba8_image() {
        let gray = image::GrayAlphaImage::from_raw(2, 1, vec![200, 128, 9, 0]).unwrap();
        let rgba = image::RgbaImage::from_raw(2, 1, vec![200, 200, 200, 128, 9, 9, 9, 0]).unwrap();
        assert_eq!(
            NemotronOmniProcessor::rgb_on_white(&DynamicImage::from(gray)),
            NemotronOmniProcessor::rgb_on_white(&DynamicImage::from(rgba))
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
        let out = NemotronOmniProcessor::new().video_patches(&refs).unwrap();
        // Two tubelets (the third frame pairs with itself) of a 32x32 grid.
        assert_eq!(out.feature_token_counts, [2 * 32 * 32 / 4]);
        let values = out.encoder_input.as_f32();
        let rows_per_frame = values.shape()[0] / 3;
        let first_value = |frame: usize| values[[frame * rows_per_frame, 0]];
        assert!(first_value(0) < first_value(1) && first_value(1) < first_value(2));
    }

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

    /// The Reference processor of a checkpoint.
    fn processor(kind: &str) -> NemotronOmniProcessor {
        NemotronOmniProcessor::reference(
            NemotronGeometry::from_model_config(&config(kind)).unwrap(),
        )
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
        // The Packed processor plans with the geometry its checkpoint's config reads as.
        assert_eq!(omni, NemotronOmniProcessor::new().geometry);
        let sup = processor("super").geometry;
        assert_eq!(
            (sup.video_frame_size, sup.video_temporal_patch_size),
            (VideoFrameSize::Square, 2)
        );

        let mut sized = config("omni");
        sized["vision_config"]["video_target_num_patches"] = Value::Null;
        sized["vision_config"]["video_target_img_size"] = json!(500);
        assert_eq!(
            NemotronGeometry::from_model_config(&sized)
                .unwrap()
                .video_frame_size,
            VideoFrameSize::Patches {
                patches: 32 * 32,
                keep_aspect: true
            }
        );
        sized["vision_config"]["video_target_num_patches"] = json!(1024);
        assert_eq!(
            NemotronGeometry::from_model_config(&sized),
            Err(NemotronConfigError::Invalid(
                "/vision_config/video_target_img_size"
            ))
        );
        let mut broken = config("v2");
        broken.as_object_mut().unwrap().remove("norm_std");
        assert_eq!(
            NemotronGeometry::from_model_config(&broken),
            Err(NemotronConfigError::Missing("/norm_std"))
        );
        // A std the writer cannot divide by, and a maximum that is no whole
        // patch count, are refused when the config is read.
        broken["norm_std"] = json!([0.0, 1.0, 1.0]);
        assert_eq!(
            NemotronGeometry::from_model_config(&broken),
            Err(NemotronConfigError::Invalid("/norm_std"))
        );
        let mut fractional = config("omni");
        fractional["vision_config"]["args"]["max_num_patches"] = json!(0.5);
        assert_eq!(
            NemotronGeometry::from_model_config(&fractional),
            Err(NemotronConfigError::Invalid(
                "/vision_config/args/max_num_patches"
            ))
        );
    }

    /// Reference dynamic images: one flat patch batch whatever their sizes,
    /// with each image's resized grid, token count and patch count.
    #[test]
    fn sizes_dynamic_images_into_one_flat_batch() {
        let processor = processor("omni");
        let images = [
            &DynamicImage::new_rgb8(640, 480),
            &DynamicImage::new_rgb8(1920, 1080),
        ];
        let context = |token_budget| VisionPreprocessingContext { token_budget };
        let out = processor
            .preprocess_with_context(&images, &context(None))
            .unwrap();
        assert_eq!(out.encoder_input.shape(), &[9360, 3 * 16 * 16]);
        assert_eq!(out.feature_token_counts, [300, 2040]);
        assert_eq!(
            ints(out.model_specific.get("imgs_sizes")),
            vec![480, 640, 1088, 1920]
        );
        assert_eq!(
            ints(out.model_specific.get("num_tokens_per_image")),
            vec![300, 2040]
        );
        assert_eq!(
            ints(out.model_specific.get("image_num_patches")),
            vec![1200, 8160]
        );
        let budgeted = processor
            .preprocess_with_context(&images[..1], &context(Some(303)))
            .unwrap();
        assert_eq!(budgeted.encoder_input.shape(), &[1140, 3 * 16 * 16]);
        // Two images of one size also join into one flat batch.
        let joined = processor
            .preprocess_with_context(&[images[0], images[0]], &context(None))
            .unwrap();
        assert_eq!(joined.encoder_input.shape(), &[2400, 3 * 16 * 16]);
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

    /// Reference: a clip's frames in order, with their source indices,
    /// duration and tubelet tokens.
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
