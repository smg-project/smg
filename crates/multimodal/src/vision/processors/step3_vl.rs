//! The Step3-VL image processor: a global view and sliding-window crops.
//!
//! The reference is the serving engine's own processor module for the
//! family. The checkpoints ship no `preprocessor_config.json`; the engine
//! builds the processor from its constants and reads one switch from the
//! model config, `vision_config.enable_patch`. Per image:
//!
//! 1. an image under 32 px on its short side with an aspect ratio past 4 is
//!    pasted at the top-left of a black square of its long side;
//! 2. a long side over 3024 px is scaled down to 3024 (both sides truncated to
//!    whole pixels) with Pillow's BILINEAR resize; the result is the global
//!    view's source;
//! 3. the crop window: a long side under 728 px takes crops of the short side
//!    when the aspect ratio is over 1.5 and none otherwise; from 728 px the
//!    window is 504, or the short side when that is under 504 and the aspect
//!    ratio over 4;
//! 4. with a window, each side at least one window long is rounded to a
//!    multiple of the window (up when the fraction is over 0.2), the view is
//!    resized to that size (Pillow BILINEAR) and cut into windows row by row,
//!    the last window of a side pulled back to its edge; a side shorter than
//!    the window keeps its length and the window reads black past it;
//! 5. the global view and every crop go through torchvision's `ToTensor` and
//!    `Normalize` (CLIP's mean and std) and its antialiased BILINEAR tensor
//!    resize to 728 x 728 and 504 x 504 (a tensor already that size passes
//!    through).
//!
//! The rule bounds nothing but the long side. The serving engine publishes
//! 3164 prompt tokens as its per-image maximum for the family (the layout of
//! a 3024 x 3024 image, the size it names as the one with the most features:
//! 36 crops, five newlines, the view; 3085 of those positions carry a
//! feature) and admits a larger item only under
//! its batched-token budget, a deployment flag (2048, 8192 or 16384 by the
//! engine's memory size at the pin). A thin image (an aspect ratio past 4
//! with a short side of 32 px or more, one under 32 px after the cap
//! included) takes a window as small as its short side, with the crop count
//! growing as the long side over it: a 2000 x 40 image plans 50 crops and
//! 4321 prompt tokens, a 96768 x 32 image 3024 one-pixel windows and 8.6 GiB of
//! float32. A plan past the published maximum, like a side that caps to
//! zero, is refused before any pixel moves, with the size, the crops, the
//! count and the limit named: strips thinner than about 1:36 that an engine
//! with a larger budget would serve are refused here, and on an engine at the
//! 2048 default the same strips never schedule (a stall the gateway cannot
//! see), so the bound is the gateway's own admission line, like the
//! per-request image count; inside it every image is the engine's bit for
//! bit. The square pad is the one allocation the rule makes before the cap,
//! and it has the image's own long side: a 100000 x 31 image pads a
//! 100000 x 100000 square (30 GB) that caps to the 3024 view and counts the
//! published maximum exactly. The square's bytes are held to the allocation
//! limit the loader reserves every decoded image against (512 MiB): a square
//! over 13377 px is refused before any pixel moves, with the square, its
//! bytes and the limit named, and a 10000 x 10 image's 300 MB square is
//! inside it, as before.
//!
//! The output carries the global views as the encoder input
//! `[N, 3, 728, 728]`, the crops as `patch_pixel_values` `[P, 3, 504, 504]`
//! (flat, split per image by `num_patches`), `num_patches` `[N]`, and
//! `patch_newline_mask` `[P]`, one per crop, set after every crop row but the
//! last. The prompt layout per image is `crops * (81 + 2) + 169 + 2 +
//! newlines` tokens: the crops each between a start and an end token, a
//! newline token after every row but the last, then the view between its own
//! start and end tokens. The feature count per image is the image tokens
//! inside that layout, `crops * 81 + 169`: the rows the encoder produces and
//! the engine scatters into those positions. The start, end and newline
//! tokens around them are prompt text, as the other tile families' are.

use std::{borrow::Cow, collections::HashMap};

use image::{DynamicImage, RgbImage};
use ndarray::Array4;
use serde_json::Value;

use super::llava::{CLIP_MEAN, CLIP_STD};
use crate::vision::{
    processor::{ModelSpecificValue, PreprocessedEncoderInputs, VisionPreProcessor},
    scratch,
    transforms::{self, TransformError},
};

/// The global view's side after the tensor resize.
pub const IMAGE_SIZE: u32 = 728;
/// A crop's side after the tensor resize.
pub const PATCH_SIZE: u32 = 504;
/// The longest side a decoded image is scaled down to before anything else.
const MAX_IMAGE_SIZE: u32 = 3024;
/// The feature tokens of the global view.
pub const IMAGE_FEATURE_TOKENS: usize = 169;
/// The feature tokens of one crop.
pub const PATCH_FEATURE_TOKENS: usize = 81;
/// The per-image maximum the serving engine publishes for the family: the
/// prompt layout of a 3024 x 3024 image (36 crops, five newlines, the view),
/// structural tokens included. A plan past it is refused; see the module doc.
pub const MAX_IMAGE_TOKENS: usize = 3164;

/// The crops, flat across the batch: `[P, 3, 504, 504]`.
pub const PATCH_PIXEL_VALUES: &str = "patch_pixel_values";
/// The crop count per image: `[N]`.
pub const NUM_PATCHES: &str = "num_patches";
/// One flag per crop, set after every crop row but the last: `[P]`.
pub const PATCH_NEWLINE_MASK: &str = "patch_newline_mask";

/// The crop window of one image and the grid it cuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropGrid {
    /// The window's side in the source the crops are cut from.
    pub window: u32,
    /// The size the view is resized to before the cut; a side under the
    /// window keeps the view's length.
    pub source: (u32, u32),
    /// Windows along the width.
    pub columns: usize,
    /// Windows along the height.
    pub rows: usize,
}

/// What the family's rule makes of one decoded size, before any pixel moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropPlan {
    /// The side of the black square the image is pasted onto when its short
    /// side is under 32 px and its aspect ratio past 4; `None` otherwise.
    pub square: Option<u32>,
    /// The global view's source size: the (padded) image under the 3024 px
    /// cap, as `(width, height)`.
    pub view: (u32, u32),
    /// The crops' window and grid; `None` when the image takes no crops.
    pub crops: Option<CropGrid>,
}

impl CropPlan {
    /// The crops of the image.
    pub fn num_patches(&self) -> usize {
        self.crops.map_or(0, |grid| grid.columns * grid.rows)
    }

    /// The newline tokens between the crop rows.
    pub fn num_newlines(&self) -> usize {
        self.crops.map_or(0, |grid| grid.rows - 1)
    }

    /// The tokens one image expands to: every crop between its start and end
    /// tokens, a newline after every row but the last, the view between its
    /// own start and end tokens.
    pub fn num_tokens(&self) -> usize {
        self.num_patches() * (PATCH_FEATURE_TOKENS + 2)
            + IMAGE_FEATURE_TOKENS
            + 2
            + self.num_newlines()
    }

    /// The positions of that expansion that carry an encoder feature: the
    /// image tokens of every crop and of the view. The start, end and newline
    /// tokens around them are prompt text with no feature behind them.
    pub fn num_features(&self) -> usize {
        self.num_patches() * PATCH_FEATURE_TOKENS + IMAGE_FEATURE_TOKENS
    }
}

/// The square pad: the side of the square an image under 32 px on its short
/// side with an aspect ratio past 4 is pasted onto, `None` for any other.
fn square_side(width: u32, height: u32) -> Option<u32> {
    let ratio = f64::from(width) / f64::from(height);
    (width.min(height) < 32 && (ratio > 4.0 || ratio < 0.25)).then(|| width.max(height))
}

/// The size under the 3024 px cap: both sides scaled by the cap over the long
/// side and truncated, as the reference's `int()` truncates them.
fn capped_size(width: u32, height: u32) -> (u32, u32) {
    let long = width.max(height);
    if long <= MAX_IMAGE_SIZE {
        return (width, height);
    }
    let scale = f64::from(MAX_IMAGE_SIZE) / f64::from(long);
    (
        (f64::from(width) * scale) as u32,
        (f64::from(height) * scale) as u32,
    )
}

/// The crop window of a view, `0` for none.
fn window_size(long: u32, short: u32) -> u32 {
    let ratio = f64::from(long) / f64::from(short);
    if long < IMAGE_SIZE {
        if ratio > 1.5 {
            short
        } else {
            0
        }
    } else if ratio > 4.0 {
        short.min(PATCH_SIZE)
    } else {
        PATCH_SIZE
    }
}

/// One side of the size the view is cut at: a whole number of windows, rounded
/// up when the fraction is over 0.2; a side under the window keeps its length.
fn crop_axis(length: u32, window: u32) -> u32 {
    let ratio = f64::from(length) / f64::from(window);
    if ratio < 1.0 {
        return length;
    }
    let whole = length / window;
    let fraction = ratio - f64::from(whole);
    window * if fraction > 0.2 { whole + 1 } else { whole }
}

/// The window starts along one side: one per window length, the last pulled
/// back so that it ends at the edge; a side under the window has one window
/// at 0.
fn window_starts(length: u32, window: u32) -> Vec<u32> {
    let count = if length <= window {
        1
    } else {
        (f64::from(length - window) / f64::from(window) + 1.0).ceil() as u32
    };
    let mut starts: Vec<u32> = (0..count).map(|i| window * i).collect();
    if let Some(last) = starts.last_mut() {
        if count > 1 && *last + window > length {
            *last = length - window;
        }
    }
    starts
}

/// The Step3-VL image processor.
#[derive(Debug, Clone)]
pub struct Step3VLProcessor {
    /// Whether the image takes crops beside its global view
    /// (`vision_config.enable_patch`, true when the config says nothing).
    enable_patch: bool,
    /// Every byte through `ToTensor` then `Normalize` with CLIP's mean and std,
    /// per channel.
    table: [[f32; 256]; 3],
}

impl Default for Step3VLProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Step3VLProcessor {
    /// The processor as the reference builds it from its constants, crops on.
    pub fn new() -> Self {
        Self::with_crops(true)
    }

    /// The processor with crops on or off.
    pub fn with_crops(enable_patch: bool) -> Self {
        Self {
            enable_patch,
            table: transforms::to_tensor_then_normalize_table(&CLIP_MEAN, &CLIP_STD),
        }
    }

    /// The processor as the reference builds it for a checkpoint: its
    /// constants, and `vision_config.enable_patch` from the model config
    /// (crops on when the config says nothing). The family ships no
    /// preprocessor config, and the reference reads none.
    pub fn from_model_config(config: &Value) -> Self {
        let enable_patch = config
            .get("vision_config")
            .and_then(|vision| vision.get("enable_patch"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        Self::with_crops(enable_patch)
    }

    /// Whether images take crops beside their global view.
    pub fn crops_enabled(&self) -> bool {
        self.enable_patch
    }

    /// The plan of a `width` x `height` image: the pad, the view and the crop
    /// grid the reference's rule derives from the size alone.
    pub fn plan(&self, width: u32, height: u32) -> CropPlan {
        let square = square_side(width, height);
        let (padded_w, padded_h) = square.map_or((width, height), |side| (side, side));
        let view = capped_size(padded_w, padded_h);
        let (long, short) = (view.0.max(view.1), view.0.min(view.1));
        let window = if self.enable_patch {
            window_size(long, short)
        } else {
            0
        };
        let crops = (window > 0).then(|| {
            let source = (crop_axis(view.0, window), crop_axis(view.1, window));
            CropGrid {
                window,
                source,
                columns: window_starts(source.0, window).len(),
                rows: window_starts(source.1, window).len(),
            }
        });
        CropPlan {
            square,
            view,
            crops,
        }
    }

    /// The plan the processor carries out for a `width` x `height` image: a
    /// zero side (decoded, or after the cap), a square pad over the image
    /// decoder's allocation limit and a plan past [`MAX_IMAGE_TOKENS`] are
    /// refused here, before any pixel moves, so the token count and the
    /// preprocessing agree on every image (the gateway's own admission line;
    /// see the module doc).
    fn checked_plan(&self, width: u32, height: u32) -> Result<CropPlan, TransformError> {
        if width == 0 || height == 0 {
            return Err(TransformError::ShapeError(format!(
                "a {width}x{height} image has no view to resize"
            )));
        }
        let plan = self.plan(width, height);
        if let Some(side) = plan.square {
            // The square is pasted in full before the resize, as the reference
            // pastes it, with the image's own long side: it is held to the
            // allocation limit the loader reserves every decoded image against
            // (`image::Limits::default()`, the decoder's), so a strip's long
            // side can ask for no more memory than a decoded image may.
            let bytes = u64::from(side)
                .saturating_mul(u64::from(side))
                .saturating_mul(3);
            if let Some(limit) = image::Limits::default()
                .max_alloc
                .filter(|limit| *limit < bytes)
            {
                return Err(TransformError::ShapeError(format!(
                    "a {width}x{height} image pads a {side}x{side} square of {bytes} bytes before its resize, above the {limit} bytes the image decoder admits for one decoded image"
                )));
            }
        }
        if plan.view.0 == 0 || plan.view.1 == 0 {
            return Err(TransformError::ShapeError(format!(
                "a {width}x{height} image is {}x{} under the {MAX_IMAGE_SIZE} px cap: no side to resize",
                plan.view.0, plan.view.1
            )));
        }
        let tokens = plan.num_tokens();
        if tokens > MAX_IMAGE_TOKENS {
            return Err(TransformError::ShapeError(format!(
                "a {width}x{height} image plans {} crops and {tokens} prompt tokens, above the {MAX_IMAGE_TOKENS} the serving engine publishes as its per-image maximum (a {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE} image's; a larger item is admitted only under its batched-token budget, a deployment setting)",
                plan.num_patches()
            )));
        }
        Ok(plan)
    }

    /// `rgb` through the table and torchvision's antialiased BILINEAR resize
    /// to `side` x `side`, appended to `planes` as `[3, side, side]`; an image
    /// already that size goes through the table alone, as the reference's
    /// resize returns such a tensor unchanged.
    fn normalized_square(
        &self,
        rgb: &RgbImage,
        side: u32,
        planes: &mut Vec<f32>,
    ) -> Result<(), TransformError> {
        let (width, height) = rgb.dimensions();
        if (width, height) == (side, side) {
            let (pixels, _) = rgb.as_raw().as_chunks::<3>();
            for (channel, table) in self.table.iter().enumerate() {
                planes.extend(
                    pixels
                        .iter()
                        .map(|pixel| table[usize::from(pixel[channel])]),
                );
            }
        } else {
            let resized = transforms::resize_bilinear_torch_aa_normalized(
                rgb.as_raw(),
                width,
                height,
                side,
                side,
                &self.table,
            )?;
            let (pixels, _) = resized.as_chunks::<3>();
            for channel in 0..3 {
                planes.extend(pixels.iter().map(|pixel| pixel[channel]));
            }
        }
        Ok(())
    }

    /// One image through the reference's steps under its checked `plan`: its
    /// view's planes come back, its crops' planes and newline flags are
    /// appended to the batch's `crops` and `newlines`.
    fn process(
        &self,
        image: &DynamicImage,
        plan: CropPlan,
        crops: &mut Vec<f32>,
        newlines: &mut Vec<i64>,
    ) -> Result<Vec<f32>, TransformError> {
        let source: Cow<'_, RgbImage> = match image {
            DynamicImage::ImageRgb8(rgb) => Cow::Borrowed(rgb),
            other => Cow::Owned(other.to_rgb8()),
        };
        let black = [0_u8; 3];
        let padded = match plan.square {
            Some(side) => Cow::Owned(transforms::paste_rgb(&source, side, side, 0, 0, black)),
            None => source,
        };
        let view = if padded.dimensions() == plan.view {
            padded
        } else {
            Cow::Owned(transforms::resize_bilinear_pil_rgb(
                padded.as_raw(),
                padded.width(),
                padded.height(),
                plan.view.0,
                plan.view.1,
            )?)
        };
        let side = IMAGE_SIZE as usize;
        let mut view_planes = scratch::take_f32_cap(3 * side * side);
        self.normalized_square(&view, IMAGE_SIZE, &mut view_planes)?;

        if let Some(grid) = plan.crops {
            let cut = if view.dimensions() == grid.source {
                Cow::Borrowed(view.as_ref())
            } else {
                Cow::Owned(transforms::resize_bilinear_pil_rgb(
                    view.as_raw(),
                    view.width(),
                    view.height(),
                    grid.source.0,
                    grid.source.1,
                )?)
            };
            let columns = window_starts(grid.source.0, grid.window);
            let rows = window_starts(grid.source.1, grid.window);
            for (row, &y) in rows.iter().enumerate() {
                for (column, &x) in columns.iter().enumerate() {
                    let crop = transforms::crop_rgb(&cut, x, y, grid.window, grid.window);
                    // A window past the edge of a side shorter than it reads
                    // black there, as the reference's crop does.
                    let crop = if crop.dimensions() == (grid.window, grid.window) {
                        crop
                    } else {
                        transforms::paste_rgb(&crop, grid.window, grid.window, 0, 0, black)
                    };
                    self.normalized_square(&crop, PATCH_SIZE, crops)?;
                    newlines.push(i64::from(
                        column + 1 == columns.len() && row + 1 < rows.len(),
                    ));
                }
            }
        }
        Ok(view_planes)
    }
}

impl VisionPreProcessor for Step3VLProcessor {
    /// Each image's pad, resizes, crops and normalization read that image
    /// alone, and the batch's tensors are the images' back to back.
    fn supports_per_image_preprocessing(&self) -> bool {
        true
    }

    fn default_mean(&self) -> [f64; 3] {
        CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let side = IMAGE_SIZE as usize;
        let view_len = 3 * side * side;
        let patch = PATCH_SIZE as usize;
        // Every plan is checked, and the batch's crops buffer sized from
        // them, before any pixel moves: a refused image costs nothing, and
        // every crop's planes are written into the buffer once.
        let plans = images
            .iter()
            .map(|image| self.checked_plan(image.width(), image.height()))
            .collect::<Result<Vec<_>, _>>()?;
        let total_crops: usize = plans.iter().map(CropPlan::num_patches).sum();
        let mut crops = if total_crops == 0 {
            Vec::new()
        } else {
            scratch::take_f32_cap(total_crops * 3 * patch * patch)
        };
        let mut newlines = Vec::with_capacity(total_crops);
        let mut views: Vec<Vec<f32>> = Vec::with_capacity(images.len());
        let mut counts = Vec::with_capacity(images.len());
        let mut features = Vec::with_capacity(images.len());
        let mut sizes = Vec::with_capacity(images.len());
        for (image, plan) in images.iter().zip(&plans) {
            views.push(self.process(image, *plan, &mut crops, &mut newlines)?);
            counts.push(plan.num_patches() as i64);
            features.push(plan.num_features());
            sizes.push((image.width(), image.height()));
        }
        // One image's planes are the batch; several are copied back to back.
        let data = if views.len() == 1 {
            views.swap_remove(0)
        } else {
            let mut data = Vec::with_capacity(views.len() * view_len);
            for view in views {
                data.extend_from_slice(&view);
                scratch::give_f32(view);
            }
            data
        };
        let encoder_input = Array4::from_shape_vec((images.len(), 3, side, side), data)
            .map_err(|error| TransformError::ShapeError(error.to_string()))?;
        let model_specific = HashMap::from([
            (
                PATCH_PIXEL_VALUES.to_string(),
                ModelSpecificValue::Tensor {
                    data: crops,
                    shape: vec![total_crops, 3, patch, patch],
                },
            ),
            (
                NUM_PATCHES.to_string(),
                ModelSpecificValue::IntTensor {
                    data: counts,
                    shape: vec![images.len()],
                },
            ),
            (
                PATCH_NEWLINE_MASK.to_string(),
                ModelSpecificValue::IntTensor {
                    data: newlines,
                    shape: vec![total_crops],
                },
            ),
        ]);
        Ok(PreprocessedEncoderInputs {
            encoder_input: encoder_input.into(),
            feature_token_counts: features,
            item_sizes: sizes,
            model_specific,
        })
    }

    /// The features of a `width` x `height` image, the count `preprocess`
    /// reports for it: the image tokens of its layout, not the layout's length.
    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        Ok(self.checked_plan(width, height)?.num_features())
    }

    fn model_name(&self) -> &'static str {
        "step3_vl"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        Some((IMAGE_SIZE, IMAGE_SIZE))
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};
    use serde_json::json;

    use super::*;

    fn plan(width: u32, height: u32) -> CropPlan {
        Step3VLProcessor::new().plan(width, height)
    }

    /// The features of the engine's most-features image: 36 crops and the view.
    const MAX_FEATURES: usize = 36 * PATCH_FEATURE_TOKENS + IMAGE_FEATURE_TOKENS;

    /// The reference's crop counts, newline counts and token counts, and the
    /// feature counts behind them, over the sizes that exercise every branch
    /// of its rule: no crops under a 728 px
    /// long side at a gentle aspect ratio, crops of the short side past 1.5,
    /// the 504 window from 728 px, the short-side window past an aspect ratio
    /// of 4, the square pad under 32 px, and the 3024 px cap.
    #[test]
    fn the_plan_counts_the_references_crops_newlines_and_tokens() {
        for (width, height, crops, newlines, tokens) in [
            (1, 1, 0, 0, 171),
            (7, 5, 0, 0, 171),
            (64, 48, 0, 0, 171),
            (640, 480, 0, 0, 171),
            (727, 500, 0, 0, 171),
            (728, 728, 4, 1, 504),
            (1024, 768, 4, 1, 504),
            (1920, 1080, 8, 1, 836),
            (600, 300, 2, 0, 337),
            (1000, 300, 2, 0, 337),
            (2000, 40, 50, 0, 4321),
            (40, 2000, 50, 49, 4370),
            (4000, 15, 36, 5, 3164),
            (10000, 10, 36, 5, 3164),
            (3, 900, 4, 1, 504),
            (4096, 2304, 24, 3, 2166),
            (3840, 2160, 24, 3, 2166),
            (8192, 8192, 36, 5, 3164),
        ] {
            let plan = plan(width, height);
            assert_eq!(plan.num_patches(), crops, "{width}x{height} crops");
            assert_eq!(plan.num_newlines(), newlines, "{width}x{height} newlines");
            assert_eq!(plan.num_tokens(), tokens, "{width}x{height} tokens");
            let features = crops * PATCH_FEATURE_TOKENS + IMAGE_FEATURE_TOKENS;
            assert_eq!(plan.num_features(), features, "{width}x{height} features");
            // The public entry point counts the features of a plan inside the
            // bound and refuses past it (2000x40 and 40x2000 here).
            let counted = Step3VLProcessor::new().calculate_num_tokens(width, height);
            if tokens > MAX_IMAGE_TOKENS {
                assert!(counted.is_err(), "{width}x{height} refused");
            } else {
                assert_eq!(counted.unwrap(), features, "{width}x{height} counted");
            }
        }
    }

    /// The sizes behind the counts: the square pad and the cap of a 4000x15
    /// strip (a 3024 square cut in 6 x 6), the second resize of a 1920x1080
    /// frame to 2016x1008, a 1000x300 image whose 300 px side stays under the
    /// window, and a 2000x40 strip cut in 40 px windows.
    #[test]
    fn the_plan_derives_the_references_sizes() {
        assert_eq!(
            plan(4000, 15),
            CropPlan {
                square: Some(4000),
                view: (3024, 3024),
                crops: Some(CropGrid {
                    window: 504,
                    source: (3024, 3024),
                    columns: 6,
                    rows: 6,
                }),
            }
        );
        assert_eq!(
            plan(1920, 1080),
            CropPlan {
                square: None,
                view: (1920, 1080),
                crops: Some(CropGrid {
                    window: 504,
                    source: (2016, 1008),
                    columns: 4,
                    rows: 2,
                }),
            }
        );
        assert_eq!(
            plan(1000, 300),
            CropPlan {
                square: None,
                view: (1000, 300),
                crops: Some(CropGrid {
                    window: 504,
                    source: (1008, 300),
                    columns: 2,
                    rows: 1,
                }),
            }
        );
        assert_eq!(
            plan(2000, 40),
            CropPlan {
                square: None,
                view: (2000, 40),
                crops: Some(CropGrid {
                    window: 40,
                    source: (2000, 40),
                    columns: 50,
                    rows: 1,
                }),
            }
        );
        assert_eq!(
            plan(8192, 8192),
            CropPlan {
                square: None,
                view: (3024, 3024),
                crops: Some(CropGrid {
                    window: 504,
                    source: (3024, 3024),
                    columns: 6,
                    rows: 6,
                }),
            }
        );
        assert_eq!(plan(640, 480).crops, None);
    }

    /// The window starts of a side: one per window with the last pulled back
    /// to the edge, and one window at 0 for a side under the window.
    #[test]
    fn window_starts_follow_the_references_sliding_window() {
        assert_eq!(window_starts(1008, 504), vec![0, 504]);
        assert_eq!(window_starts(1000, 504), vec![0, 496]);
        assert_eq!(window_starts(300, 504), vec![0]);
        assert_eq!(window_starts(504, 504), vec![0]);
        assert_eq!(window_starts(2000, 40).len(), 50);
    }

    /// A model config switches the crops off; the view stays.
    #[test]
    fn the_model_config_switches_the_crops_off() {
        let processor =
            Step3VLProcessor::from_model_config(&json!({"vision_config": {"enable_patch": false}}));
        assert!(!processor.crops_enabled());
        let plan = processor.plan(1920, 1080);
        assert_eq!(plan.crops, None);
        assert_eq!(plan.num_tokens(), 171);
        assert_eq!(plan.num_features(), IMAGE_FEATURE_TOKENS);
        assert!(Step3VLProcessor::from_model_config(&json!({"vision_config": {}})).crops_enabled());
    }

    /// The tensors of a 1000x300 image: one view, two crops of which the
    /// second's right part and both crops' lower parts read the black past the
    /// image (the normalized zero byte), one newline flag of zero.
    #[test]
    fn a_short_side_under_the_window_reads_black_past_the_image() {
        let processor = Step3VLProcessor::new();
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(1000, 300, Rgb([200, 100, 50])));
        let out = processor.preprocess(&[&image]).unwrap();
        assert_eq!(out.encoder_input.shape(), &[1, 3, 728, 728]);
        assert_eq!(
            out.feature_token_counts,
            vec![2 * PATCH_FEATURE_TOKENS + IMAGE_FEATURE_TOKENS]
        );
        assert_eq!(out.item_sizes, vec![(1000, 300)]);
        let Some(ModelSpecificValue::Tensor { data, shape }) =
            out.model_specific.get(PATCH_PIXEL_VALUES)
        else {
            panic!("the crops tensor");
        };
        assert_eq!(shape, &[2, 3, 504, 504]);
        let black = processor.table[0][0];
        let crop_len = 3 * 504 * 504;
        // The first crop's red plane: the image's colour in its upper 300
        // rows, black below.
        let first = &data[..crop_len];
        assert_eq!(first[0], processor.table[0][200]);
        assert_eq!(first[299 * 504], processor.table[0][200]);
        assert_eq!(first[300 * 504], black);
        assert_eq!(first[503 * 504 + 503], black);
        assert!(matches!(
            out.model_specific.get(NUM_PATCHES),
            Some(ModelSpecificValue::IntTensor { data, shape }) if data == &[2] && shape == &[1]
        ));
        assert!(matches!(
            out.model_specific.get(PATCH_NEWLINE_MASK),
            Some(ModelSpecificValue::IntTensor { data, shape }) if data == &[0, 0] && shape == &[2]
        ));
    }

    /// Two images of a batch stack their views and lay their crops back to
    /// back, with the per-image counts and the newline flags in crop order.
    #[test]
    fn a_batch_lays_the_views_and_crops_back_to_back() {
        let processor = Step3VLProcessor::new();
        let small = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 48, Rgb([10, 20, 30])));
        let frame = DynamicImage::ImageRgb8(RgbImage::from_pixel(1920, 1080, Rgb([40, 50, 60])));
        let out = processor.preprocess(&[&small, &frame]).unwrap();
        assert_eq!(out.encoder_input.shape(), &[2, 3, 728, 728]);
        assert_eq!(
            out.feature_token_counts,
            vec![
                IMAGE_FEATURE_TOKENS,
                8 * PATCH_FEATURE_TOKENS + IMAGE_FEATURE_TOKENS
            ]
        );
        assert!(matches!(
            out.model_specific.get(NUM_PATCHES),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &[0, 8]
        ));
        assert!(matches!(
            out.model_specific.get(PATCH_NEWLINE_MASK),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &[0, 0, 0, 1, 0, 0, 0, 0]
        ));
        assert!(matches!(
            out.model_specific.get(PATCH_PIXEL_VALUES),
            Some(ModelSpecificValue::Tensor { shape, .. }) if shape == &[8, 3, 504, 504]
        ));
        let single = processor.preprocess(&[&frame]).unwrap();
        let both = out.encoder_input.flat_f32();
        assert_eq!(&both[3 * 728 * 728..], &single.encoder_input.flat_f32()[..]);
    }

    /// An image with no crops still carries the crops tensor, empty, as the
    /// reference emits it, so the engine takes the pixel path.
    #[test]
    fn an_image_without_crops_carries_an_empty_crops_tensor() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(640, 480, Rgb([1, 2, 3])));
        let out = Step3VLProcessor::new().preprocess(&[&image]).unwrap();
        assert!(matches!(
            out.model_specific.get(PATCH_PIXEL_VALUES),
            Some(ModelSpecificValue::Tensor { data, shape }) if data.is_empty() && shape == &[0, 3, 504, 504]
        ));
        assert!(matches!(
            out.model_specific.get(PATCH_NEWLINE_MASK),
            Some(ModelSpecificValue::IntTensor { data, shape }) if data.is_empty() && shape == &[0]
        ));
    }

    /// An empty batch and a zero-sized image are refused.
    #[test]
    fn an_empty_batch_and_a_zero_sized_image_are_refused() {
        let processor = Step3VLProcessor::new();
        assert!(matches!(
            processor.preprocess(&[]),
            Err(TransformError::EmptyBatch)
        ));
        assert!(processor.calculate_num_tokens(0, 10).is_err());
    }

    #[test]
    fn a_plan_past_the_engines_declared_image_is_refused_before_any_pixel_moves() {
        let processor = Step3VLProcessor::new();
        // The engine's own count: a 3024 x 3024 image, 36 crops, five newlines;
        // 3085 of its 3164 positions carry a feature.
        assert_eq!(processor.plan(3024, 3024).num_tokens(), MAX_IMAGE_TOKENS);
        assert_eq!(
            processor.calculate_num_tokens(3024, 3024).unwrap(),
            MAX_FEATURES
        );
        // One row of 36 windows of 84 px: just under the count, the engine's own plan.
        assert_eq!(processor.plan(3024, 84).num_tokens(), MAX_IMAGE_TOKENS - 5);
        assert_eq!(
            processor.calculate_num_tokens(3024, 84).unwrap(),
            MAX_FEATURES
        );
        // 6048 x 64: the short side passes the pad, the cap halves it to 32, 95 windows.
        let error = processor
            .calculate_num_tokens(6048, 64)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("6048x64")
                && error.contains("95 crops")
                && error.contains("8056 prompt tokens")
                && error.contains("3164"),
            "{error}"
        );
        // The edge: 96768 x 32 caps to 3024 x 1, 3024 windows of one pixel.
        let error = processor
            .calculate_num_tokens(96768, 32)
            .unwrap_err()
            .to_string();
        assert!(error.contains("3024 crops"), "{error}");
        // A side that caps to zero has nothing to resize.
        let error = processor
            .calculate_num_tokens(100_000, 32)
            .unwrap_err()
            .to_string();
        assert!(error.contains("3024x0"), "{error}");
        // The preprocessing refuses the same images, before any tensor is built.
        let thin = DynamicImage::new_rgb8(6048, 64);
        assert!(processor.preprocess(&[&thin]).is_err());
        let edge = DynamicImage::new_rgb8(96_768, 32);
        assert!(processor.preprocess(&[&edge]).is_err());
    }

    #[test]
    fn a_thin_images_square_pad_is_held_to_the_decoders_allocation_limit() {
        let processor = Step3VLProcessor::new();
        // The budget is the decoder's own: the limit every decoded image is
        // reserved against.
        let limit = image::Limits::default().max_alloc.unwrap();
        assert_eq!(limit, 512 * 1024 * 1024);
        // The accepted rows stay: a 10000 x 10 strip pads a 300 MB square.
        assert_eq!(processor.plan(10_000, 10).square, Some(10_000));
        assert_eq!(
            processor.calculate_num_tokens(10_000, 10).unwrap(),
            MAX_FEATURES
        );
        // The last square inside the limit (536832387 bytes) and the first
        // past it.
        assert_eq!(
            processor.calculate_num_tokens(13_377, 31).unwrap(),
            MAX_FEATURES
        );
        let error = processor
            .calculate_num_tokens(13_378, 31)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("13378x31")
                && error.contains("13378x13378 square")
                && error.contains("536912652 bytes")
                && error.contains("536870912"),
            "{error}"
        );
        // A 100000 x 31 strip pads a 30 GB square; a JPEG at the format's
        // 65535 px maximum 12.9 GB.
        let error = processor
            .calculate_num_tokens(100_000, 31)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("100000x100000 square") && error.contains("30000000000 bytes"),
            "{error}"
        );
        let error = processor
            .calculate_num_tokens(65_535, 31)
            .unwrap_err()
            .to_string();
        assert!(error.contains("12884508675 bytes"), "{error}");
        // The preprocessing refuses them before the square is allocated.
        for (width, height) in [(100_000, 31), (65_535, 31), (31, 100_000)] {
            let strip = DynamicImage::new_rgb8(width, height);
            assert!(processor.preprocess(&[&strip]).is_err(), "{width}x{height}");
        }
    }
}
