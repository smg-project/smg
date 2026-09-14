//! DeepSeek-V4.1 vision preprocessor.
//!
//! Port of the reference `load_image` in the checkpoint's
//! `inference/image_processor.py` (the same math as vLLM's
//! `models/deepseek_v4_1/common/mm_preprocess.py`) so that token counts and
//! pixel values match the reference. Each `<｜deepseek_image｜>` placeholder
//! in the prompt expands to `[IMAGE_START] + ([IMAGE] * n_llm_w +
//! [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]` positions; every span position
//! carries `image_token_id` in `input_ids` and the roles ride along in a
//! per-image `types` tensor (the reference's out-of-band `token_types`).
//! IMAGE slots receive aligner rows in reading order; the delimiters take the
//! learned `image_start` / `image_newline` / `image_end` vectors.
//!
//! The span is spliced verbatim: vLLM removed its compressor-alignment pad
//! (vllm-project/vllm#56554), so the budget is the full `max_image_tokens`
//! and no token is reserved.

use image::{DynamicImage, Rgb, RgbImage};
use ndarray::{Array2, Array4};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::VisionPreProcessor,
        transforms::{pad_to_size_pil, resize_bicubic_pil_rgb, TransformError},
    },
};

/// Span role tags carried by the per-image `types` tensor (the reference's
/// out-of-band `token_types`).
pub const IMAGE_START: i64 = 0;
pub const IMAGE: i64 = 1;
pub const IMAGE_NEW_LINE: i64 = 2;
pub const IMAGE_END: i64 = 3;

/// Padding color of the reference `ImageOps.pad` call.
const PAD_COLOR: Rgb<u8> = Rgb([127, 127, 127]);

const DEFAULT_PATCH_SIZE: usize = 14;
const DEFAULT_DOWNSAMPLE_RATIO: usize = 3;
const DEFAULT_MAX_NUM_TOKENS: usize = 1024;
/// The geometry needs a start, an end, and at least one row of one token: a
/// configured `max_image_tokens` below this would underflow the budget maths.
const MIN_MAX_NUM_TOKENS: usize = 4;
const DEFAULT_MIN_PIXELS: usize = 295936;

/// Resize plan for one image: ViT patch grid, LLM token grid, and the target
/// pixel size, all derived from the original pixel size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridPlan {
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    pub best_height: usize,
    pub best_width: usize,
}

impl GridPlan {
    /// Total number of LLM tokens the image span occupies.
    pub fn num_image_tokens(&self) -> usize {
        num_image_tokens(self.n_llm_h, self.n_llm_w)
    }
}

fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

/// Token grid the aligner produces from a patch grid of this pixel size.
fn llm_grid(
    best_height: usize,
    best_width: usize,
    patch_size: usize,
    downsample_ratio: usize,
) -> (usize, usize) {
    (
        ((best_height / patch_size) as f64 / downsample_ratio as f64).ceil() as usize,
        ((best_width / patch_size) as f64 / downsample_ratio as f64).ceil() as usize,
    )
}

/// Largest aspect-preserving pixel size whose token grid still fits in
/// max_n_token. Returns (best_height, best_width).
fn solve_resize_ratio(
    height: f64,
    width: f64,
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> (usize, usize) {
    let r = height / width;
    let max_w_float = ((max_n_token - 2) as f64 / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    let cell = patch_size * downsample_ratio;
    if max_w_float < 1.0 {
        // very tall: collapse to a single column
        return ((max_n_token - 2) / 2 * cell, cell);
    }
    if max_h_float < 1.0 {
        // very wide: collapse to a single row
        return (cell, (max_n_token - 3) * cell);
    }
    let beta =
        (max_w_float.floor() * cell as f64 / width).min(max_h_float.floor() * cell as f64 / height);
    (
        (height * beta / patch_size as f64).floor() as usize * patch_size,
        (width * beta / patch_size as f64).floor() as usize * patch_size,
    )
}

/// Shrink the pixel size until the image costs at most max_n_token LLM tokens.
fn safe_resize(
    height: f64,
    width: f64,
    best_height: usize,
    best_width: usize,
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> (usize, usize, usize, usize) {
    let (mut n_llm_h, mut n_llm_w) =
        llm_grid(best_height, best_width, patch_size, downsample_ratio);
    let (mut best_height, mut best_width) = (best_height, best_width);
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token {
        (best_height, best_width) =
            solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token);
        (n_llm_h, n_llm_w) = llm_grid(best_height, best_width, patch_size, downsample_ratio);
        debug_assert!(num_image_tokens(n_llm_h, n_llm_w) <= max_n_token);
    }
    (n_llm_h, n_llm_w, best_height, best_width)
}

/// Reading-order span layout: one IMAGE_NEW_LINE per row.
pub fn image_token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<i64> {
    let mut types = Vec::with_capacity(num_image_tokens(n_llm_h, n_llm_w));
    types.push(IMAGE_START);
    for _ in 0..n_llm_h {
        types.extend(std::iter::repeat_n(IMAGE, n_llm_w));
        types.push(IMAGE_NEW_LINE);
    }
    types.push(IMAGE_END);
    types
}

/// DeepSeek-V4.1 per-image transform (the PIL-input equivalent of the
/// reference `load_image`).
#[derive(Debug, Clone)]
pub struct DeepseekV41Processor {
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
    min_pixels: usize,
    max_wh_ratio: Option<f64>,
}

impl Default for DeepseekV41Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepseekV41Processor {
    /// Processor with the checkpoint's `vision_config` defaults.
    pub fn new() -> Self {
        Self {
            patch_size: DEFAULT_PATCH_SIZE,
            downsample_ratio: DEFAULT_DOWNSAMPLE_RATIO,
            max_n_token: DEFAULT_MAX_NUM_TOKENS,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_wh_ratio: None,
        }
    }

    /// Build from a preprocessor config, falling back to the checkpoint
    /// defaults for absent fields. DeepSeek-V4.1 ships no
    /// `preprocessor_config.json`; overrides use the `vision_config` field
    /// names from `config.json`.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        // Every geometry input is floored at 1: a zero from a malformed
        // config would divide by zero (patch) or saturate the grid
        // (downsample) instead of producing a request error later.
        Self {
            patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE).max(1),
            downsample_ratio: config
                .get_extra("downsample_ratio")
                .unwrap_or(DEFAULT_DOWNSAMPLE_RATIO)
                .max(1),
            max_n_token: config
                .get_extra("max_image_tokens")
                .unwrap_or(DEFAULT_MAX_NUM_TOKENS)
                .max(MIN_MAX_NUM_TOKENS),
            min_pixels: config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
            max_wh_ratio: config.get_extra("max_wh_ratio"),
        }
    }

    fn with_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        if config.patch_size.is_some()
            || config.min_pixels.is_some()
            || config.extra.contains_key("downsample_ratio")
            || config.extra.contains_key("max_image_tokens")
            || config.extra.contains_key("max_wh_ratio")
        {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        }
    }

    /// Resize plan for an image of the given original size; a pure function
    /// of its arguments (the reference `plan_image_grid` / `load_image`
    /// geometry).
    pub fn plan_image_grid(&self, width: u32, height: u32) -> GridPlan {
        let p = self.patch_size;
        let mut width = f64::from(width);
        let mut height = f64::from(height);
        if let Some(max_wh_ratio) = self.max_wh_ratio {
            if width > height * max_wh_ratio {
                width = height * max_wh_ratio;
            }
        }
        if 0.0 < width * height && width * height < self.min_pixels as f64 {
            let ratio = (self.min_pixels as f64 / (width * height)).sqrt();
            // Python `int()` truncates toward zero; the values are positive.
            width = (width * ratio).trunc();
            height = (height * ratio).trunc();
        }
        let best_width = (width / p as f64).ceil() as usize * p;
        let best_height = (height / p as f64).ceil() as usize * p;
        let (n_llm_h, n_llm_w, best_height, best_width) = safe_resize(
            height,
            width,
            best_height,
            best_width,
            p,
            self.downsample_ratio,
            self.max_n_token,
        );
        GridPlan {
            n_vit_h: best_height / p,
            n_vit_w: best_width / p,
            n_llm_h,
            n_llm_w,
            best_height,
            best_width,
        }
    }

    /// Transform one decoded image into normalized ViT patches.
    ///
    /// Same math as the reference `load_image`, except the image is already
    /// decoded (the gateway supplies images instead of a record dict).
    fn load_image(&self, image: &DynamicImage) -> Result<(Array2<f32>, GridPlan), TransformError> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let plan = self.plan_image_grid(width, height);
        let (out_w, out_h) = (plan.best_width as u32, plan.best_height as u32);
        let transformed = match self.max_wh_ratio {
            // The reference stretches with `>=` here while `plan_image_grid`
            // clamps the width with `>`, so at exactly
            // `width == max_wh_ratio * height` the width is kept and the
            // image is stretched rather than padded. Pillow's `Image.resize`
            // defaults to BICUBIC.
            Some(max_wh_ratio) if f64::from(width) >= max_wh_ratio * f64::from(height) => {
                resize_bicubic_pil_rgb(rgb.as_raw(), width, height, out_w, out_h)?
            }
            _ => pad_to_size_pil(&DynamicImage::ImageRgb8(rgb), out_w, out_h, PAD_COLOR)?.to_rgb8(),
        };
        Ok((self.patchify(&transformed, &plan), plan))
    }

    /// Normalize (`/ 255`, then `(x - 0.5) / 0.5`) fused with patchification:
    /// patch `(i, j)` in row-major order holds `[c, dy, dx]` elements, matching
    /// the reference `reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4)`.
    #[expect(
        clippy::expect_used,
        reason = "the patch buffer is sized from the plan it is reshaped by"
    )]
    fn patchify(&self, image: &RgbImage, plan: &GridPlan) -> Array2<f32> {
        let p = self.patch_size;
        let n_patches = plan.n_vit_h * plan.n_vit_w;
        let patch_len = 3 * p * p;
        let mut out = vec![0.0f32; n_patches * patch_len];
        let raw = image.as_raw();
        let row_stride = plan.best_width * 3;
        for i in 0..plan.n_vit_h {
            for j in 0..plan.n_vit_w {
                let patch = &mut out[(i * plan.n_vit_w + j) * patch_len..][..patch_len];
                for c in 0..3 {
                    for dy in 0..p {
                        for dx in 0..p {
                            // f32 op order matches the reference: v/255, then
                            // (x - 0.5) / 0.5.
                            patch[c * p * p + dy * p + dx] =
                                (raw[(i * p + dy) * row_stride + (j * p + dx) * 3 + c] as f32
                                    / 255.0
                                    - 0.5)
                                    / 0.5;
                        }
                    }
                }
            }
        }
        Array2::from_shape_vec((n_patches, patch_len), out)
            .expect("patch buffer matches its shape by construction")
    }
}

impl VisionPreProcessor for DeepseekV41Processor {
    fn default_mean(&self) -> [f64; 3] {
        [0.5, 0.5, 0.5]
    }

    fn default_std(&self) -> [f64; 3] {
        [0.5, 0.5, 0.5]
    }

    #[expect(
        clippy::expect_used,
        reason = "per-image patch arrays are standard-layout and the concatenation is sized by construction"
    )]
    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let processor = self.with_preprocessor_config(config);

        let mut patches = Vec::new();
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut vit_grid = Vec::with_capacity(images.len() * 2);
        let mut llm_grid = Vec::with_capacity(images.len() * 2);
        let mut types = Vec::new();
        let mut patches_per_image = Vec::with_capacity(images.len());
        let mut types_per_image = Vec::with_capacity(images.len());

        let mut total_patches = 0usize;
        for image in images {
            let (image_patches, plan) = processor.load_image(image)?;
            total_patches += plan.n_vit_h * plan.n_vit_w;
            patches.push(image_patches);
            feature_token_counts.push(plan.num_image_tokens());
            item_sizes.push((image.width(), image.height()));
            vit_grid.extend([plan.n_vit_h as i64, plan.n_vit_w as i64]);
            llm_grid.extend([plan.n_llm_h as i64, plan.n_llm_w as i64]);
            types.extend(image_token_types(plan.n_llm_h, plan.n_llm_w));
            patches_per_image.push((plan.n_vit_h * plan.n_vit_w) as i64);
            types_per_image.push(plan.num_image_tokens() as i64);
        }

        let patch_len = 3 * processor.patch_size * processor.patch_size;
        let mut encoder_input = Vec::with_capacity(total_patches * patch_len);
        for image_patches in &patches {
            encoder_input.extend_from_slice(
                image_patches
                    .as_slice()
                    .expect("patch array is standard-layout by construction"),
            );
        }
        // The engine consumes patches as `(np, 3, p, p)` (see vLLM's
        // `DeepseekV4VLImagePixelInputs`).
        let p = processor.patch_size;
        let encoder_input = Array4::from_shape_vec((total_patches, 3, p, p), encoder_input)
            .expect("concatenated patch buffer matches its shape by construction");

        let n_images = images.len();
        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra(
                    "vit_grid",
                    ModelSpecificValue::IntTensor {
                        data: vit_grid,
                        shape: vec![n_images, 2],
                    },
                )
                .with_extra(
                    "llm_grid",
                    ModelSpecificValue::IntTensor {
                        data: llm_grid,
                        shape: vec![n_images, 2],
                    },
                )
                .with_extra("types", ModelSpecificValue::int_1d(types))
                .with_extra(
                    "patches_per_image",
                    ModelSpecificValue::int_1d(patches_per_image),
                )
                .with_extra(
                    "types_per_image",
                    ModelSpecificValue::int_1d(types_per_image),
                ),
        )
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        self.with_preprocessor_config(config)
            .plan_image_grid(width, height)
            .num_image_tokens()
    }

    fn model_name(&self) -> &'static str {
        "deepseek_v41"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(width: u32, height: u32) -> GridPlan {
        DeepseekV41Processor::new().plan_image_grid(width, height)
    }

    /// Reference values from the checkpoint's `image_processor.py`
    /// (patch=14, downsample=3, max_n_token=1024, min_pixels=295936,
    /// max_wh_ratio=None) and DeepSeek's `deepseek-recipe` token spec; the
    /// budget is the full 1024 tokens since no alignment pad is reserved.
    #[test]
    fn plan_matches_reference_table_without_pad_reservation() {
        let cases = [
            ((100, 100), (546, 546), (13, 13), 184),
            ((640, 480), (644, 490), (16, 12), 206),
            ((1024, 768), (1036, 770), (25, 19), 496),
            ((1024, 1024), (1036, 1036), (25, 25), 652),
            ((1920, 1080), (1708, 966), (41, 23), 968),
            ((4000, 4000), (1302, 1302), (31, 31), 994),
            ((4000, 3000), (1512, 1134), (36, 27), 1001),
            ((8000, 1000), (3696, 462), (88, 11), 981),
        ];
        for ((w, h), (bw, bh), (lw, lh), tokens) in cases {
            let p = plan(w, h);
            assert_eq!((p.best_width, p.best_height), (bw, bh), "{w}x{h}");
            assert_eq!((p.n_llm_w, p.n_llm_h), (lw, lh), "{w}x{h}");
            assert_eq!(p.num_image_tokens(), tokens, "{w}x{h}");
        }
        // Small images are upscaled to min_pixels (a square of 544x544 =
        // 295936 pixels); ceil-to-patch makes the best size 546x546.
        assert_eq!((plan(100, 100).n_vit_w, plan(100, 100).n_vit_h), (39, 39));
        assert_eq!(
            (plan(4000, 4000).n_vit_w, plan(4000, 4000).n_vit_h),
            (93, 93)
        );
    }

    #[test]
    fn extreme_aspects_collapse_to_single_strip_using_the_full_budget() {
        // Extremely tall: solve_resize_ratio collapses to a single column of
        // 511 rows (511 * 2 + 2 = 1024).
        let p = plan(2, 1_000_000);
        assert_eq!((p.n_llm_w, p.n_llm_h), (1, 511));
        assert_eq!(p.num_image_tokens(), 1024);

        // Extremely wide: collapses to a single row of 1021 tokens
        // (1 * 1022 + 2 = 1024).
        let p = plan(1_000_000, 2);
        assert_eq!((p.n_llm_w, p.n_llm_h), (1021, 1));
        assert_eq!(p.num_image_tokens(), 1024);
    }

    #[test]
    fn token_types_are_reading_order_with_row_delimiters() {
        assert_eq!(
            image_token_types(2, 3),
            vec![
                IMAGE_START,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE_END,
            ]
        );
    }

    /// `max_wh_ratio` (absent from the checkpoint, so no golden covers it)
    /// selects the reference's stretch arm with `>=` while the plan clamps
    /// with `>`: at and above the ratio a solid image fills every patch (a
    /// contain-fit would leave gray pad rows); below it the pad arm runs.
    #[test]
    fn max_wh_ratio_selects_the_stretch_arm_like_the_reference() {
        let config: PreProcessorConfig =
            serde_json::from_value(serde_json::json!({"max_wh_ratio": 2.0})).unwrap();
        let processor = DeepseekV41Processor::new().with_preprocessor_config(&config);
        let unbounded = DeepseekV41Processor::new();
        assert_eq!(processor.max_wh_ratio, Some(2.0));

        // Wider than the ratio: the plan clamps the width first; exactly the
        // ratio: not clamped.
        assert!(
            processor.plan_image_grid(300, 100).best_width
                < unbounded.plan_image_grid(300, 100).best_width
        );
        assert_eq!(
            processor.plan_image_grid(200, 100).best_width,
            unbounded.plan_image_grid(200, 100).best_width
        );

        let white = Rgb([255, 255, 255]);
        let fill_values = |w: u32, h: u32| -> Vec<f32> {
            let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, white));
            let out = processor
                .preprocess(std::slice::from_ref(&image), &config)
                .unwrap();
            out.encoder_input.iter().copied().collect()
        };
        for (w, h) in [(300, 100), (200, 100)] {
            assert!(
                fill_values(w, h).iter().all(|v| (v - 1.0).abs() < 1e-6),
                "{w}x{h} must be stretched, not padded"
            );
        }
        // 160x100 is below the ratio and its best box is not exactly 1.6:1,
        // so the contain-fit pads gray columns.
        assert!(
            fill_values(160, 100).iter().any(|v| *v < 0.5),
            "160x100 must be padded, not stretched"
        );
    }

    #[test]
    fn respects_max_image_tokens() {
        let processor = DeepseekV41Processor::new();
        let config: PreProcessorConfig =
            serde_json::from_value(serde_json::json!({"max_image_tokens": 320})).unwrap();

        assert!(processor.calculate_num_tokens(4000, 4000, &config) <= 320);
        assert!(processor.calculate_num_tokens(4000, 4000, &PreProcessorConfig::default()) > 320);
        // A budget below the geometry's minimum is floored instead of
        // underflowing the resize maths.
        let tiny: PreProcessorConfig =
            serde_json::from_value(serde_json::json!({"max_image_tokens": 1})).unwrap();
        assert_eq!(processor.calculate_num_tokens(4000, 4000, &tiny), 4);
    }

    #[test]
    fn extreme_aspect_images_error_instead_of_panicking() {
        let processor = DeepseekV41Processor::new();
        let config = PreProcessorConfig::default();
        let tall = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 50_000, Rgb([0, 0, 0])));
        assert!(matches!(
            processor.preprocess(std::slice::from_ref(&tall), &config),
            Err(TransformError::ShapeError(_))
        ));
        let wide = DynamicImage::ImageRgb8(RgbImage::from_pixel(100_000, 1, Rgb([0, 0, 0])));
        assert!(matches!(
            processor.preprocess(std::slice::from_ref(&wide), &config),
            Err(TransformError::ShapeError(_))
        ));
    }

    #[test]
    fn preprocess_emits_engine_contract() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 48, Rgb([10, 200, 30])));
        let processor = DeepseekV41Processor::new();
        let config = PreProcessorConfig::default();
        let out = processor
            .preprocess(std::slice::from_ref(&image), &config)
            .unwrap();

        let plan = processor.plan_image_grid(64, 48);
        assert_eq!((plan.best_width, plan.best_height), (630, 476));
        let n_vit = plan.n_vit_h * plan.n_vit_w;
        let p = processor.patch_size;
        assert_eq!(out.encoder_input.shape(), &[n_vit, 3, p, p]);
        assert_eq!(out.feature_token_counts, vec![plan.num_image_tokens()]);
        assert_eq!(out.item_sizes, vec![(64, 48)]);

        let ModelSpecificValue::IntTensor { data, shape } = &out.model_specific["vit_grid"] else {
            panic!("vit_grid must be an int tensor");
        };
        assert_eq!(shape, &[1, 2]);
        assert_eq!(data, &[plan.n_vit_h as i64, plan.n_vit_w as i64]);

        let ModelSpecificValue::IntTensor { data, .. } = &out.model_specific["patches_per_image"]
        else {
            panic!("patches_per_image must be an int tensor");
        };
        assert_eq!(data, &[n_vit as i64]);

        let ModelSpecificValue::IntTensor { data: types, .. } = &out.model_specific["types"] else {
            panic!("types must be an int tensor");
        };
        assert_eq!(types.len(), plan.num_image_tokens());
        assert_eq!(types.first(), Some(&IMAGE_START));
        assert_eq!(types.last(), Some(&IMAGE_END));

        // The image is contain-fit into a slightly taller canvas, so the
        // top-left pixel is the gray pad color ...
        let pad_value = (127.0f32 / 255.0 - 0.5) / 0.5;
        assert!((out.encoder_input[[0, 0, 0, 0]] - pad_value).abs() < 1e-6);
        // ... while a center patch carries the (R=10) image content.
        let center_patch = (plan.n_vit_h / 2) * plan.n_vit_w + plan.n_vit_w / 2;
        let expected = (10.0f32 / 255.0 - 0.5) / 0.5;
        assert!((out.encoder_input[[center_patch, 0, 0, 0]] - expected).abs() < 0.02);
    }
}
