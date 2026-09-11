//! DeepSeek V4 Flash Vision image preprocessing.
//!
//! A faithful Rust port of `inference/image_processor.py` in
//! `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`: the resize-solver that fits an
//! image into the aligner token budget, the patch grid extraction, and the
//! N-layout sentinel block (`types` + `perm`) the engine's aligner consumes.
//!
//! Layout contract with the serving engine (see
//! `tokenspeed/runtime/models/deepseek_v4_vl.py`):
//! - `encoder_input` : `[total_patches, 3, patch, patch]` f32, normalized
//!   `(x / 255 - 0.5) / 0.5`; the engine casts to bf16.
//! - `feature_token_counts[i]` : `n_llm_h * n_llm_w` IMAGE slots of item `i`.
//! - `model_specific["n_vit_h"] / ["n_vit_w"]` : per-item ViT grid rows/cols.
//! - `model_specific["types"]` : per-item sentinel blocks, **concatenated**;
//!   each block starts with the maximal `COMPRESS_PAD_TO - 1` leading pads.
//! - `model_specific["perm"]`   : per-item aligner row order, concatenated.
//! - `model_specific["types_lengths"]` : per-item `types` lengths for slicing.
//! - `model_specific["patch_counts"]` / `["perm_lengths"]` : per-item ViT
//!   patch and aligner-row counts for slicing the concatenated tensors.
//!
//! The leading compress pads are position-dependent in the official flow
//! (`3 - block_start % 4`); the replacement layer trims
//! `block_start % 4` leading pads from each expanded token sequence, and the
//! engine applies the same trim to `types` (see `DeepseekV4Spec`).

use image::{DynamicImage, GenericImageView};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    vision::{
        preprocessor_config::PreProcessorConfig, processor::VisionPreProcessor,
        transforms::TransformError,
    },
};

/// Sentinel token types, mirroring the official
/// `IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)`.
pub const IMAGE_START: i64 = 0;
pub const IMAGE_PAD: i64 = 1;
pub const IMAGE: i64 = 2;
pub const IMAGE_NEW_LINE: i64 = 3;
pub const IMAGE_END: i64 = 4;

/// The aligner's compression alignment: IMAGE data must start at a token
/// index divisible by this.
pub const COMPRESS_PAD_TO: usize = 4;

const DEFAULT_PATCH_SIZE: usize = 14;
const DEFAULT_DOWNSAMPLE_RATIO: usize = 3;
const DEFAULT_MAX_N_TOKEN: usize = 384;
const DEFAULT_MIN_PIXELS: usize = 147_456;
const DEFAULT_MAX_WH_RATIO: Option<usize> = Some(8);

#[derive(Debug, Clone, Copy)]
struct Params {
    patch: usize,
    downsample: usize,
    max_n_token: usize,
    min_pixels: usize,
    max_wh_ratio: Option<usize>,
}

fn extra_usize(
    config: &PreProcessorConfig,
    key: &str,
    default: usize,
) -> Result<usize, TransformError> {
    match config.extra.get(key) {
        None => Ok(default),
        Some(value) => value.as_u64().map(|v| v as usize).ok_or_else(|| {
            TransformError::ShapeError(format!(
                "deepseek_v4 preprocessor key {key} must be a non-negative integer, got {value}"
            ))
        }),
    }
}

impl Params {
    fn from_config(config: &PreProcessorConfig) -> Result<Self, TransformError> {
        let max_wh_ratio = match config.extra.get("vision_max_wh_ratio") {
            None => DEFAULT_MAX_WH_RATIO,
            Some(value) => value.as_u64().map(|v| Some(v as usize)).ok_or_else(|| {
                TransformError::ShapeError(format!(
                    "deepseek_v4 preprocessor key vision_max_wh_ratio must be a non-negative integer, got {value}"
                ))
            })?,
        };
        Ok(Self {
            patch: extra_usize(config, "vision_patch_size", DEFAULT_PATCH_SIZE)?,
            downsample: extra_usize(config, "vision_downsample_ratio", DEFAULT_DOWNSAMPLE_RATIO)?,
            max_n_token: extra_usize(config, "vision_max_n_token", DEFAULT_MAX_N_TOKEN)?,
            min_pixels: extra_usize(config, "vision_min_pixels", DEFAULT_MIN_PIXELS)?,
            max_wh_ratio,
        })
    }
}

/// Number of LLM tokens the aligner grid occupies (N-layout, incl. padding).
/// Direct port of the official `grid_tokens`.
fn grid_tokens(best_h: usize, best_w: usize, p: usize, r: usize) -> (usize, usize, usize) {
    let n_llm_h = (best_h / p).div_ceil(r);
    let n_llm_w = (best_w / p).div_ceil(r);
    let row_len = n_llm_w + 1;
    let mut num_tokens = n_llm_h * row_len + 2;
    if n_llm_h % 2 == 1 {
        num_tokens += row_len;
    }
    num_tokens += (n_llm_h.div_ceil(2) * row_len % 2) * 2;
    (n_llm_h, n_llm_w, num_tokens)
}

/// Port of the official `solve_resize_ratio`: pick the largest even-row
/// aligner grid whose token count fits `max_n_token`, preserving aspect.
fn solve_resize_ratio(
    height: usize,
    width: usize,
    p: usize,
    r: usize,
    max_n_token: usize,
) -> (usize, usize, usize, usize, usize) {
    let ratio = height as f64 / width as f64;
    let max_w_float = ((max_n_token as f64 - 2.0) / ratio + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * ratio;
    let (best_w, best_h);
    if max_w_float < 1.0 {
        let max_w = 1usize;
        let mut max_h = (max_n_token - 2) / (max_w + 1);
        if max_h % 2 == 1 {
            max_h -= 1;
        }
        best_w = max_w * p * r;
        best_h = max_h * p * r;
    } else if max_h_float < 2.0 {
        let max_h = 2usize;
        let max_w = (max_n_token - 2) / max_h - 1;
        if max_w <= 1 {
            return (0, 0, 0, 0, usize::MAX); // caller keeps shrinking; guard below
        }
        best_w = max_w * p * r;
        best_h = max_h * p * r;
    } else {
        let mut max_h = max_h_float.floor() as usize;
        let max_w = max_w_float.floor() as usize;
        if max_h % 2 == 1 {
            max_h -= 1;
        }
        let beta = (max_w * p * r) as f64 / width as f64;
        let beta = beta.min((max_h * p * r) as f64 / height as f64);
        best_w = ((width as f64 * beta) as usize / p) * p;
        best_h = ((height as f64 * beta) as usize / p) * p;
    }
    let (n_llm_h, n_llm_w, num) = grid_tokens(best_h, best_w, p, r);
    (n_llm_h, n_llm_w, best_h, best_w, num)
}

/// Port of the official `safe_resize`: shrink the token budget until the
/// grid fits. Returns `(n_llm_h, n_llm_w, best_h, best_w)`.
fn safe_resize(
    height: usize,
    width: usize,
    best_h: usize,
    best_w: usize,
    p: usize,
    r: usize,
    max_n_token: usize,
) -> (usize, usize, usize, usize) {
    let cap = max_n_token.saturating_sub(COMPRESS_PAD_TO - 1);
    let (n_llm_h, n_llm_w, mut num) = grid_tokens(best_h, best_w, p, r);
    let mut budget = cap;
    let mut geom = (n_llm_h, n_llm_w, best_h, best_w);
    while num > cap {
        let (h, w, bh, bw, n) = solve_resize_ratio(height, width, p, r, budget);
        if n == usize::MAX || bh == 0 || bw == 0 {
            // Degenerate solve (budget collapsed below the smallest legal
            // grid): retry once at the floor budget so the loop terminates.
            let (h, w, bh, bw, _) = solve_resize_ratio(height, width, p, r, COMPRESS_PAD_TO + 4);
            return (h, w, bh, bw);
        }
        geom = (h, w, bh, bw);
        num = n;
        budget = budget.saturating_sub(1);
    }
    geom
}

/// Per-image geometry + pixel plan after the official resize constraints.
struct PlannedImage {
    /// Resized/padded RGB canvas, `best_h * best_w * 3` (HWC, u8).
    canvas: Vec<u8>,
    best_h: usize,
    best_w: usize,
    n_vit_h: usize,
    n_vit_w: usize,
    n_llm_h: usize,
    n_llm_w: usize,
}

fn plan_image(img: &DynamicImage, params: &Params) -> Result<PlannedImage, TransformError> {
    let (iw, ih) = img.dimensions();
    let (iw, ih) = (iw as usize, ih as usize);
    if iw == 0 || ih == 0 {
        return Err(TransformError::ShapeError(
            "deepseek_v4 got a zero-sized image".to_string(),
        ));
    }
    let p = params.patch;

    // Constrain the aspect ratio first (width clamp), then enforce the
    // minimum-pixel floor — same order as the official loader.
    let mut width = iw;
    let mut height = ih;
    if let Some(ratio) = params.max_wh_ratio {
        if width > height * ratio {
            width = height * ratio;
        }
    }
    if width * height < params.min_pixels {
        let scale = (params.min_pixels as f64 / (width * height) as f64).sqrt();
        width = (width as f64 * scale) as usize;
        height = (height as f64 * scale) as usize;
        width = width.max(1);
        height = height.max(1);
    }

    let mut best_w = width.div_ceil(p) * p;
    let mut best_h = height.div_ceil(p) * p;
    let (n_llm_h, n_llm_w, rh, rw) = safe_resize(
        height,
        width,
        best_h,
        best_w,
        p,
        params.downsample,
        params.max_n_token,
    );
    best_h = rh;
    best_w = rw;
    let n_vit_h = best_h / p;
    let n_vit_w = best_w / p;

    let wide = params.max_wh_ratio.is_some_and(|ratio| iw >= ratio * ih);
    let rgb = img.to_rgb8();
    let canvas = if wide {
        // Extreme aspect: plain stretch to the target canvas.
        resize_hwc(&rgb, iw, ih, best_w, best_h)
    } else {
        // Contain: scale to fit, then pad the remainder with neutral gray.
        pad_hwc(&rgb, iw, ih, best_w, best_h, 127)
    };
    Ok(PlannedImage {
        canvas,
        best_h,
        best_w,
        n_vit_h,
        n_vit_w,
        n_llm_h,
        n_llm_w,
    })
}

/// Bilinear-free nearest resize is NOT what PIL uses; the official path uses
/// PIL's default (bicubic-ish) resize only for the stretched wide case and
/// `ImageOps.pad` (which letterboxes with a solid color) otherwise.
fn resize_hwc(src: &image::RgbImage, sw: usize, sh: usize, tw: usize, th: usize) -> Vec<u8> {
    if sw == tw && sh == th {
        return src.as_raw().clone();
    }
    // Bilinear resampling, RGB channels.
    let sx = sw as f64 / tw as f64;
    let sy = sh as f64 / th as f64;
    let mut out = Vec::with_capacity(tw * th * 3);
    for y in 0..th {
        let fy = (y as f64 + 0.5) * sy - 0.5;
        let y0 = fy.floor().clamp(0.0, (sh - 1) as f64) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = ((fy - y0 as f64).clamp(0.0, 1.0)) as f32;
        for x in 0..tw {
            let fx = (x as f64 + 0.5) * sx - 0.5;
            let x0 = fx.floor().clamp(0.0, (sw - 1) as f64) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = ((fx - x0 as f64).clamp(0.0, 1.0)) as f32;
            for c in 0..3 {
                let a = src.get_pixel(x0 as u32, y0 as u32)[c] as f32;
                let b = src.get_pixel(x1 as u32, y0 as u32)[c] as f32;
                let d = src.get_pixel(x0 as u32, y1 as u32)[c] as f32;
                let e = src.get_pixel(x1 as u32, y1 as u32)[c] as f32;
                let top = a + (b - a) * wx;
                let bot = d + (e - d) * wx;
                out.push((top + (bot - top) * wy).round().clamp(0.0, 255.0) as u8);
            }
        }
    }
    out
}

/// Contain-scale `src` into `tw x th` and letterbox the remainder with
/// `fill` on the right/bottom — `ImageOps.pad` semantics.
fn pad_hwc(src: &image::RgbImage, sw: usize, sh: usize, tw: usize, th: usize, fill: u8) -> Vec<u8> {
    let scale = (tw as f64 / sw as f64).min(th as f64 / sh as f64);
    let nw = ((sw as f64 * scale).round() as usize).clamp(1, tw);
    let nh = ((sh as f64 * scale).round() as usize).clamp(1, th);
    let inner = resize_hwc(src, sw, sh, nw, nh);
    let mut out = vec![fill; tw * th * 3];
    for y in 0..nh {
        let dst = (y * tw * 3)..(y * tw * 3 + nw * 3);
        let src_range = (y * nw * 3)..((y + 1) * nw * 3);
        out[dst].copy_from_slice(&inner[src_range]);
    }
    out
}

/// Extract the patch grid as `[n_vit_h * n_vit_w, 3, p, p]` f32 with the
/// official `(x/255 - 0.5) / 0.5` normalization, in row-major patch order.
fn extract_patches(plan: &PlannedImage, patch: usize) -> Vec<f32> {
    let PlannedImage {
        canvas,
        n_vit_h,
        n_vit_w,
        ..
    } = plan;
    let count = n_vit_h * n_vit_w;
    let mut out = Vec::with_capacity(count * 3 * patch * patch);
    // Permute equivalent: reshape(3, nh, p, nw, p).permute(1,3,0,2,4)
    // → iterate patches (gy, gx), then channel, then (py, px).
    for gy in 0..*n_vit_h {
        for gx in 0..*n_vit_w {
            for c in 0..3 {
                for py in 0..patch {
                    for px in 0..patch {
                        let h = gy * patch + py;
                        let w = gx * patch + px;
                        let v = canvas[(h * *n_vit_w * patch + w) * 3 + c] as f32;
                        out.push((v / 255.0 - 0.5) / 0.5);
                    }
                }
            }
        }
    }
    out
}

/// Build the N-layout sentinel block for one image at nominal position 0.
/// Returns `(types, perm)`:
/// - `types`: the full token-type sequence, starting with the maximal
///   `COMPRESS_PAD_TO - 1` leading pads (the replacement layer trims
///   `block_start % 4` of them at splice time);
/// - `perm`: aligner-row order of the IMAGE slots.
///
/// Port of the official `build_image_block` with `start_pos = 0`.
pub fn build_image_block(n_llm_h: usize, n_llm_w: usize) -> (Vec<i64>, Vec<i64>) {
    let pad_h = n_llm_h % 2;
    let rows = n_llm_h + pad_h;
    let row_len = n_llm_w + 1;
    let pad_last = (rows / 2 * row_len % 2) * 2;

    // Sentinel grid before interleaving: IMAGE rows with a trailing newline,
    // plus a full pad row when n_llm_h is odd.
    let mut core = Vec::with_capacity(rows * row_len);
    let mut image_pos = vec![-1i64; rows * row_len];
    for r in 0..rows {
        for c in 0..row_len {
            core.push(if r < n_llm_h {
                if c < n_llm_w {
                    IMAGE
                } else {
                    IMAGE_NEW_LINE
                }
            } else {
                IMAGE_PAD
            });
        }
    }
    for r in 0..n_llm_h {
        for c in 0..n_llm_w {
            image_pos[r * row_len + c] = (r * n_llm_w + c) as i64;
        }
    }

    let mut types = Vec::with_capacity(rows * row_len + COMPRESS_PAD_TO + 2);
    types.extend(std::iter::repeat_n(IMAGE_PAD, COMPRESS_PAD_TO - 1));
    types.push(IMAGE_START);
    let mut perm = Vec::with_capacity(n_llm_h * n_llm_w);
    // Interleave row pairs column-major: view(rows/2, 2, L).transpose(1,2)
    // enumerates (pair, col, k), i.e. rows 2p and 2p+1 zipped by column.
    for pair in 0..rows / 2 {
        for c in 0..row_len {
            for k in 0..2 {
                let idx = (pair * 2 + k) * row_len + c;
                types.push(core[idx]);
                if image_pos[idx] >= 0 {
                    perm.push(image_pos[idx]);
                }
            }
        }
    }
    types.extend(std::iter::repeat_n(IMAGE_PAD, pad_last));
    types.push(IMAGE_END);
    (types, perm)
}

pub struct DeepseekV4Processor;

impl VisionPreProcessor for DeepseekV4Processor {
    fn default_mean(&self) -> [f64; 3] {
        [0.5, 0.5, 0.5]
    }

    fn default_std(&self) -> [f64; 3] {
        [0.5, 0.5, 0.5]
    }

    fn model_name(&self) -> &'static str {
        "deepseek_v4"
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let params = match Params::from_config(config) {
            Ok(p) => p,
            Err(_) => return 0,
        };
        let mut w = width as usize;
        let h = height as usize;
        if let Some(ratio) = params.max_wh_ratio {
            if w > h * ratio {
                w = h * ratio;
            }
        }
        let bw = w.div_ceil(params.patch) * params.patch;
        let bh = h.div_ceil(params.patch) * params.patch;
        let (_, _, num) = grid_tokens(bh, bw, params.patch, params.downsample);
        num.min(params.max_n_token)
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let params = Params::from_config(config)?;
        if params.patch == 0 || params.downsample == 0 {
            return Err(TransformError::ShapeError(
                "deepseek_v4 patch size and downsample ratio must be positive".to_string(),
            ));
        }

        let mut patches_flat: Vec<f32> = Vec::new();
        let mut feature_token_counts: Vec<usize> = Vec::with_capacity(images.len());
        let mut item_sizes: Vec<(u32, u32)> = Vec::with_capacity(images.len());
        let mut n_vit_h_vec: Vec<i64> = Vec::with_capacity(images.len());
        let mut n_vit_w_vec: Vec<i64> = Vec::with_capacity(images.len());
        let mut types_flat: Vec<i64> = Vec::new();
        let mut perm_flat: Vec<i64> = Vec::new();
        let mut types_lengths: Vec<i64> = Vec::with_capacity(images.len());
        let mut patch_counts: Vec<i64> = Vec::with_capacity(images.len());
        let mut perm_lengths: Vec<i64> = Vec::with_capacity(images.len());

        for img in images {
            let plan = plan_image(img, &params)?;
            let (types, perm) = build_image_block(plan.n_llm_h, plan.n_llm_w);
            patches_flat.extend(extract_patches(&plan, params.patch));
            feature_token_counts.push(plan.n_llm_h * plan.n_llm_w);
            item_sizes.push((plan.best_w as u32, plan.best_h as u32));
            n_vit_h_vec.push(plan.n_vit_h as i64);
            n_vit_w_vec.push(plan.n_vit_w as i64);
            types_lengths.push(types.len() as i64);
            patch_counts.push((plan.n_vit_h * plan.n_vit_w) as i64);
            perm_lengths.push(perm.len() as i64);
            types_flat.extend(types);
            perm_flat.extend(perm);
        }

        let shape = vec![
            patch_counts.iter().sum::<i64>() as usize,
            3,
            params.patch,
            params.patch,
        ];
        let encoder_input = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), patches_flat)
            .map_err(|_| {
            TransformError::ShapeError("deepseek_v4 patch tensor shape mismatch".to_string())
        })?;

        let perm_len = perm_flat.len();
        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra("n_vit_h", ModelSpecificValue::IntVec(n_vit_h_vec))
                .with_extra("n_vit_w", ModelSpecificValue::IntVec(n_vit_w_vec))
                .with_extra(
                    "types",
                    ModelSpecificValue::IntTensor {
                        data: types_flat,
                        shape: vec![types_lengths.iter().sum::<i64>() as usize],
                    },
                )
                .with_extra(
                    "perm",
                    ModelSpecificValue::IntTensor {
                        data: perm_flat,
                        shape: vec![perm_len],
                    },
                )
                .with_extra("types_lengths", ModelSpecificValue::IntVec(types_lengths))
                .with_extra("patch_counts", ModelSpecificValue::IntVec(patch_counts))
                .with_extra("perm_lengths", ModelSpecificValue::IntVec(perm_lengths)),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    /// Mirrors the official Python expected outputs (computed once against
    /// inference/image_processor.py and pinned here).
    #[test]
    fn grid_tokens_matches_official() {
        // grid_tokens(756, 756, 14, 3): n_llm_h = n_llm_w = 18, rows even.
        let (h, w, n) = grid_tokens(756, 756, 14, 3);
        assert_eq!((h, w), (18, 18));
        // 18*19 + 2 = 344; +pad_last(9*19%2=1 → 2) = 346.
        assert_eq!(n, 346);

        // Odd rows pick up a full pad row: n_llm_h=19 → rows=20.
        let (h, w, n) = grid_tokens(798, 756, 14, 3);
        assert_eq!((h, w), (19, 18));
        // 19*19 + 2 = 363; +19 (pad row) = 382; pad_last(10*19%2=0) → 382.
        assert_eq!(n, 382);
    }

    #[test]
    fn build_image_block_layout() {
        // Small grid: n_llm_h=2, n_llm_w=2 → rows=2, row_len=3.
        let (types, perm) = build_image_block(2, 2);
        // core = [I I N | I I N], no pad row; interleaved column-major pairs.
        // order = (r0c0, r1c0, r0c1, r1c1, r0c2, r1c2)
        //       = ( I,   I,   I,   I,   N,   N )
        // pad_last = (1*3 % 2) * 2 = 2.
        assert_eq!(
            types,
            vec![
                IMAGE_PAD,
                IMAGE_PAD,
                IMAGE_PAD,
                IMAGE_START,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE_NEW_LINE,
                IMAGE_PAD,
                IMAGE_PAD,
                IMAGE_END
            ]
        );
        assert_eq!(perm, vec![0, 2, 1, 3]);
    }

    #[test]
    fn build_image_block_odd_rows() {
        let (types, perm) = build_image_block(3, 1);
        // Official ground truth (inference/image_processor.py, start_pos=0):
        // rows=4 (one pad row), row_len=2, pad_last = (2*2 % 2)*2 = 0.
        // core rows: [I N | I N | I N | P P]; column-major pair interleave:
        // r0c0 r1c0 r0c1 r1c1 r2c0 r3c0 r2c1 r3c1 = I I N N I P N P.
        assert_eq!(
            types,
            vec![
                IMAGE_PAD,
                IMAGE_PAD,
                IMAGE_PAD,
                IMAGE_START,
                IMAGE,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE_NEW_LINE,
                IMAGE,
                IMAGE_PAD,
                IMAGE_NEW_LINE,
                IMAGE_PAD,
                IMAGE_END
            ]
        );
        // IMAGE positions in interleaved order: 0, 2, 4.
        assert_eq!(perm, vec![0, 1, 2]);
    }

    #[test]
    fn safe_resize_fits_budget() {
        // A huge square image must come back within the 384-token budget.
        let (n_llm_h, _, bh, bw) = safe_resize(4096, 4096, 4096, 4096, 14, 3, 384);
        let (_, _, num) = grid_tokens(bh, bw, 14, 3);
        assert!(num <= 384 - 3, "num tokens {num} exceeds budget");
        assert!(n_llm_h % 2 == 0);
        assert_eq!(bh % 14, 0);
        assert_eq!(bw % 14, 0);
    }

    #[test]
    fn safe_resize_preserves_grid_within_budget() {
        assert_eq!(safe_resize(84, 84, 84, 84, 14, 3, 384), (2, 2, 84, 84));
    }

    #[test]
    fn preprocess_png_batch_preserves_patch_and_aligner_counts() {
        let mut images = Vec::new();
        for (width, height, color) in [(84, 84, [0, 127, 255]), (126, 42, [255, 0, 127])] {
            let original = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                width,
                height,
                image::Rgb(color),
            ));
            let mut png = Cursor::new(Vec::new());
            original
                .write_to(&mut png, image::ImageFormat::Png)
                .unwrap();
            images.push(image::load_from_memory(png.get_ref()).unwrap());
        }
        let config: PreProcessorConfig = serde_json::from_value(serde_json::json!({
            "vision_min_pixels": 0
        }))
        .unwrap();
        let output = DeepseekV4Processor.preprocess(&images, &config).unwrap();
        assert_eq!(output.encoder_input.shape(), &[63, 3, 14, 14]);
        assert_eq!(output.feature_token_counts, vec![4, 3]);
        for (key, expected) in [
            ("n_vit_h", vec![6, 3]),
            ("n_vit_w", vec![6, 9]),
            ("patch_counts", vec![36, 27]),
            ("perm_lengths", vec![4, 3]),
            ("types_lengths", vec![13, 13]),
        ] {
            assert!(
                matches!(
                    output.model_specific.get(key),
                    Some(ModelSpecificValue::IntVec(actual)) if *actual == expected
                ),
                "{key}"
            );
        }
        let ModelSpecificValue::IntTensor { data: types, .. } = &output.model_specific["types"]
        else {
            panic!("missing types");
        };
        let ModelSpecificValue::IntTensor { data: perm, .. } = &output.model_specific["perm"]
        else {
            panic!("missing perm");
        };
        assert_eq!(types.iter().filter(|&&t| t == IMAGE).count(), 7);
        assert_eq!(perm, &[0, 2, 1, 3, 0, 1, 2]);
        // Patch boundaries must preserve each decoded image's normalized color.
        assert_eq!(output.encoder_input[[0, 0, 0, 0]], -1.0);
        assert_eq!(output.encoder_input[[35, 2, 13, 13]], 1.0);
        assert_eq!(output.encoder_input[[36, 0, 0, 0]], 1.0);
        assert_eq!(output.encoder_input[[62, 1, 13, 13]], -1.0);

        // The checkpoint defaults exercise the minimum-pixel upscaling path.
        let default_output = DeepseekV4Processor
            .preprocess(&images[..1], &PreProcessorConfig::default())
            .unwrap();
        assert_eq!(default_output.encoder_input.shape(), &[784, 3, 14, 14]);
        assert_eq!(default_output.feature_token_counts, vec![100]);
    }

    #[test]
    fn patch_extraction_shape_and_normalization() {
        let plan = PlannedImage {
            canvas: vec![0u8; 28 * 28 * 3],
            best_h: 28,
            best_w: 28,
            n_vit_h: 2,
            n_vit_w: 2,
            n_llm_h: 1,
            n_llm_w: 1,
        };
        let out = extract_patches(&plan, 14);
        assert_eq!(out.len(), 4 * 3 * 14 * 14);
        // All-zero canvas normalizes to -1.0.
        assert!((out[0] - (-1.0)).abs() < 1e-6);
    }
}
