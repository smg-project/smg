//! Shared MoonViT preprocessing core for the Kimi vision family.
//!
//! Kimi-K2.5 and Kimi-K3 run the same MoonViT stack:
//!
//! 1. Compute scale to fit within patch limits (never upscale)
//! 2. Resize with BICUBIC interpolation
//! 3. Zero-pad to make dimensions divisible by factor (patch_size * merge_size)
//! 4. Normalize with the configured mean/std
//! 5. Extract patches as [N, C, patch_size, patch_size]
//!
//! Kimi resizes then zero-pads to make dimensions divisible by the alignment
//! factor. The model was trained with zero-padded images, so using direct
//! resize-to-aligned would degrade image quality.
//!
//! The reference `navit_resize_image`, `navit_patchify`, and `normalize` are
//! byte-identical between the two releases. The models diverge only in their
//! configured patch budget and in whether transparent pixels are composited
//! over a background before patchify — see [`MoonVitParams`] and the
//! `transparent_bg` argument to [`preprocess`].

use std::borrow::Cow;

use image::{DynamicImage, GenericImageView};
use ndarray::Array3;

use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{ModelSpecificValue, PreprocessedEncoderInputs},
    scratch,
    transforms::{self, TransformError, TransparentBg, TransparentBgFillStage},
};

/// MoonViT resize/patchify/normalize parameters for a single model.
#[derive(Debug, Clone, Copy)]
pub struct MoonVitParams {
    pub patch_size: usize,
    pub merge_size: usize,
    /// Maximum total patches before merge (`in_patch_limit`).
    pub in_patch_limit: usize,
    /// Maximum patches along one spatial dimension.
    pub patch_limit_on_one_side: usize,
    /// Normalization mean (the config's, else CLIP's).
    pub mean: [f64; 3],
    /// Normalization std (the config's, else CLIP's).
    pub std: [f64; 3],
}

impl MoonVitParams {
    #[inline]
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Overlay whatever the model's `preprocessor_config.json` supplies.
    ///
    /// A processor resolves this once, when it is built for the model, over
    /// its compiled-in defaults. K2.5 and K3 ship different `in_patch_limit`
    /// values (16384 vs 65536), which is exactly the kind of divergence that
    /// silently caps resolution if the defaults win.
    pub fn resolved(self, config: &PreProcessorConfig) -> Self {
        Self {
            // A zero patch or merge size would divide by zero downstream.
            patch_size: config.get_patch_size(self.patch_size).max(1),
            merge_size: config.merge_size.unwrap_or(self.merge_size).max(1),
            in_patch_limit: config
                .get_extra::<usize>("in_patch_limit")
                .unwrap_or(self.in_patch_limit)
                .max(1),
            patch_limit_on_one_side: config
                .get_extra::<usize>("patch_limit_on_one_side")
                .unwrap_or(self.patch_limit_on_one_side)
                .max(1),
            mean: config.get_image_mean(),
            std: config.get_image_std(),
        }
    }

    /// Compute resize dimensions and padding, matching HF `navit_resize_image`.
    ///
    /// Never upscales (scale capped at 1.0). Pads with zeros to align to factor.
    pub fn compute_resize_config(&self, width: usize, height: usize) -> ResizeConfig {
        let ps = self.patch_size;
        let patches_w = (width / ps).max(1) as f64;
        let patches_h = (height / ps).max(1) as f64;

        let s1 = (self.in_patch_limit as f64 / (patches_w * patches_h)).sqrt();
        let s2 = (self.patch_limit_on_one_side * ps) as f64 / width as f64;
        let s3 = (self.patch_limit_on_one_side * ps) as f64 / height as f64;
        let scale = f64::min(1.0, f64::min(s1, f64::min(s2, s3)));

        let new_w = ((width as f64 * scale) as usize).max(1);
        let new_h = ((height as f64 * scale) as usize).max(1);
        let new_w = new_w.min(self.patch_limit_on_one_side * ps);
        let new_h = new_h.min(self.patch_limit_on_one_side * ps);

        let factor = self.factor();
        let pad_width = (factor - new_w % factor) % factor;
        let pad_height = (factor - new_h % factor) % factor;

        let token_height = (new_h + pad_height) / factor;
        let token_width = (new_w + pad_width) / factor;
        let num_tokens = token_height * token_width;

        ResizeConfig {
            new_width: new_w,
            new_height: new_h,
            pad_width,
            pad_height,
            num_tokens,
        }
    }
}

/// MoonViT resize configuration for a single image.
pub struct ResizeConfig {
    pub new_width: usize,
    pub new_height: usize,
    pub pad_width: usize,
    pub pad_height: usize,
    pub num_tokens: usize,
}

/// The reference's `normalize`, per channel and per 8-bit level: `x / 255.0`
/// in double precision stored as `f32`, then `x -= mean` and `x *= 1 / std`
/// with double-precision operands and an `f32` store after each step (an
/// in-place NumPy ufunc computes in the wider operand type and casts on the
/// store). A fused `x * (1 / (255 std)) - mean / std` rounds once where the
/// reference rounds three times: under the family's mean and std of 0.5 it
/// differed by one `f32` ULP on 111 of the 256 levels, which flipped the
/// bf16 rounding of 0.2-0.3 % of the values on the wire. A table of the 256
/// values per channel reproduces every rounding at one load per byte.
fn normalized_levels(mean: &[f64; 3], std: &[f64; 3]) -> [[f32; 256]; 3] {
    std::array::from_fn(|channel| {
        let std_inv = 1.0 / std[channel];
        std::array::from_fn(|level| {
            let rescaled = (level as f64 / 255.0) as f32;
            let centred = (f64::from(rescaled) - mean[channel]) as f32;
            (f64::from(centred) * std_inv) as f32
        })
    })
}

/// Fused resize + zero-pad + normalize into a single [C, H_padded, W_padded] tensor.
///
/// Avoids intermediate allocations by:
/// 1. Allocating the final padded canvas directly
/// 2. Pre-filling with normalized black (the table's entry for a zero byte,
///    as the reference zero-pads the bytes before it normalizes)
/// 3. Deinterleaving + normalizing the image region in one pass, every byte
///    through its channel's table of [`normalized_levels`]
///
/// The resize is the reference's `PIL.Image.resize(.., BICUBIC)`: Pillow's
/// separable bicubic in 22-bit fixed point (`transforms::resize_bicubic_pil_rgb`)
/// and, for an image whose alpha survives to the resize, Pillow's premultiplied
/// `RGBa` resample on the same kernel (`transforms::resize_bicubic_pil_rgba`).
/// The SIMD Catmull-Rom path of `transforms::resize` is the same filter family
/// in another quantization: it parted from the reference by one 8-bit level on
/// 12-48 % of the pixels of every image it resized. An image within the patch
/// budget keeps its size and is not resampled at all, as Pillow copies it.
///
/// Alpha handling follows the reference. With a `transparent_bg` the image is
/// composited at the configured stage; K3 ships `"after_resize"`, and the
/// ordering is load-bearing because a chessboard is generated at the resolution
/// of whatever it is painted onto. With `None` alpha is dropped *before* the
/// resize, matching the reference's `.convert("RGB")` at load time. A frame
/// carries an alpha channel here only when the gateway decoded it under
/// `TransparencyPolicy::KeepAlpha` (K3 asks for that through `keeps_alpha`;
/// otherwise the destination engine's loader rule was applied at the decode).
fn resize_pad_and_normalize(
    image: &DynamicImage,
    cfg: &ResizeConfig,
    levels: &[[f32; 256]; 3],
    transparent_bg: Option<TransparentBg>,
) -> Result<Array3<f32>, TransformError> {
    let canvas_h = cfg.new_height + cfg.pad_height;
    let canvas_w = cfg.new_width + cfg.pad_width;

    // Nothing to composite over if the image has no alpha to begin with.
    let bg = transparent_bg.filter(|_| image.color().has_alpha());

    // Reduce to RGB up front unless the background is painted after the resize,
    // in which case alpha has to survive the resample.
    let pre_flattened = match bg {
        Some(b) if b.stage == TransparentBgFillStage::BeforeResize => Some(
            DynamicImage::ImageRgb8(transforms::fill_transparent_bg(image, b.config)),
        ),
        Some(_) => None,
        None if image.color().has_alpha() => Some(DynamicImage::ImageRgb8(image.to_rgb8())),
        None => None,
    };
    let source = pre_flattened.as_ref().unwrap_or(image);

    let after_resize = bg.is_some_and(|b| b.stage == TransparentBgFillStage::AfterResize);
    let (new_w, new_h) = (cfg.new_width as u32, cfg.new_height as u32);
    let resized = if source.dimensions() == (new_w, new_h) {
        None
    } else if after_resize {
        Some(DynamicImage::ImageRgba8(
            transforms::resize_bicubic_pil_rgba(source, new_w, new_h),
        ))
    } else {
        let (width, height, rgb) = transforms::rgb_bytes(source);
        Some(DynamicImage::ImageRgb8(transforms::resize_bicubic_pil_rgb(
            &rgb,
            width as u32,
            height as u32,
            new_w,
            new_h,
        )?))
    };
    let resized = resized.as_ref().unwrap_or(source);

    let post_filled = bg
        .filter(|_| after_resize)
        .map(|b| transforms::fill_transparent_bg(resized, b.config));
    let (img_w, img_h, raw): (usize, usize, Cow<'_, [u8]>) = match &post_filled {
        Some(rgb) => (
            rgb.width() as usize,
            rgb.height() as usize,
            Cow::Borrowed(rgb.as_raw().as_slice()),
        ),
        None => transforms::rgb_bytes(resized),
    };
    let canvas_pixels = canvas_h * canvas_w;

    // Pooled: this per-image CHW buffer (tens of MB) is recycled by the
    // caller after patch extraction, keeping its pages mapped and hot.
    let mut data = scratch::take_f32(3 * canvas_pixels);
    transforms::paint_rgb_onto_planes_by_table(
        &raw,
        (img_w, img_h),
        (canvas_w, canvas_h),
        levels,
        &mut data,
    );

    #[expect(
        clippy::expect_used,
        reason = "data has exactly 3*canvas_h*canvas_w elements by construction"
    )]
    Ok(Array3::from_shape_vec((3, canvas_h, canvas_w), data)
        .expect("shape matches pre-allocated buffer"))
}

/// Extract [C, patch_size, patch_size] patches from a contiguous [C, H, W] tensor.
///
/// Uses row-based `copy_from_slice` instead of per-element indexing so the
/// compiler can auto-vectorize the inner copy.
/// Append this image's patches directly into `out` (no per-image intermediate
/// Vec): `out` is the pooled batch buffer pre-sized for the whole request.
fn extract_patches_into(tensor: &Array3<f32>, patch_size: usize, out: &mut Vec<f32>) {
    let channels = tensor.shape()[0];
    let height = tensor.shape()[1];
    let width = tensor.shape()[2];

    let grid_h = height / patch_size;
    let grid_w = width / patch_size;

    // Get contiguous slice for direct row addressing
    let flat = tensor.as_standard_layout();
    #[expect(
        clippy::expect_used,
        reason = "as_standard_layout guarantees contiguous C-order memory"
    )]
    let data = flat
        .as_slice()
        .expect("as_standard_layout guarantees contiguous memory");

    for gh in 0..grid_h {
        for gw in 0..grid_w {
            let h_start = gh * patch_size;
            let w_start = gw * patch_size;
            for c in 0..channels {
                let plane_offset = c * height * width;
                for ph in 0..patch_size {
                    let row_start = plane_offset + (h_start + ph) * width + w_start;
                    out.extend_from_slice(&data[row_start..row_start + patch_size]);
                }
            }
        }
    }
}

/// Run the full MoonViT pipeline over a batch of images.
pub fn preprocess(
    params: MoonVitParams,
    images: &[&DynamicImage],
    transparent_bg: Option<TransparentBg>,
) -> Result<PreprocessedEncoderInputs, TransformError> {
    if images.is_empty() {
        return Err(TransformError::EmptyBatch);
    }

    let item_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();
    let levels = normalized_levels(&params.mean, &params.std);

    // Pre-size the pooled batch buffer exactly (patch_features per patch =
    // 3 * patch_size^2; this is the data plane's hottest allocation).
    let patch_features = 3 * params.patch_size * params.patch_size;
    let mut estimated_total = 0usize;
    for image in images {
        let (w, h) = image.dimensions();
        let cfg = params.compute_resize_config(w as usize, h as usize);
        let grid_h = (cfg.new_height + cfg.pad_height) / params.patch_size;
        let grid_w = (cfg.new_width + cfg.pad_width) / params.patch_size;
        estimated_total += grid_h * grid_w * patch_features;
    }
    let mut all_patches: Vec<f32> = scratch::take_f32_cap(estimated_total);
    let mut patches_per_image: Vec<i64> = Vec::with_capacity(images.len());
    let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
    let mut feature_token_counts = Vec::with_capacity(images.len());

    for image in images {
        let (w, h) = image.dimensions();
        let cfg = params.compute_resize_config(w as usize, h as usize);

        // Fused resize + pad + normalize in one pass (avoids 2 extra allocations)
        let tensor = resize_pad_and_normalize(image, &cfg, &levels, transparent_bg)?;

        let padded_h = cfg.new_height + cfg.pad_height;
        let padded_w = cfg.new_width + cfg.pad_width;
        let grid_h = padded_h / params.patch_size;
        let grid_w = padded_w / params.patch_size;
        let grid_t = 1usize;

        grid_thw_data.push(grid_t as i64);
        grid_thw_data.push(grid_h as i64);
        grid_thw_data.push(grid_w as i64);

        let num_patches = grid_h * grid_w;
        feature_token_counts.push(cfg.num_tokens);

        // Patchify directly into the pooled batch buffer, then recycle the
        // CHW tensor's storage (standard layout, offset 0) for the next image.
        extract_patches_into(&tensor, params.patch_size, &mut all_patches);
        let (storage, _offset) = tensor.into_raw_vec_and_offset();
        scratch::give_f32(storage);
        patches_per_image.push(num_patches as i64);
    }

    let total_patches: usize = patches_per_image.iter().map(|&n| n as usize).sum();
    let encoder_input = ndarray::Array4::from_shape_vec(
        (total_patches, 3, params.patch_size, params.patch_size),
        all_patches,
    )
    .map_err(|e| {
        TransformError::ShapeError(format!(
            "Failed to create encoder_input [{total_patches}, 3, {}, {}]: {e}",
            params.patch_size, params.patch_size
        ))
    })?;

    Ok(
        PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
            .with_extra(
                "grid_thws",
                ModelSpecificValue::int_2d(grid_thw_data, images.len(), 3),
            )
            .with_extra(
                "patches_per_image",
                ModelSpecificValue::int_1d(patches_per_image),
            ),
    )
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::test_support::hex;

    /// The reference's three steps, each stored as `f32`.
    fn reference_level(level: u8, mean: f64, std: f64) -> f32 {
        let rescaled = (f64::from(level) / 255.0) as f32;
        let centred = (f64::from(rescaled) - mean) as f32;
        (f64::from(centred) * (1.0 / std)) as f32
    }

    /// The bits the checkpoints' own `normalize` gives a byte (NumPy 2.2,
    /// read back as `f32`): under the family's mean and std of 0.5, and under
    /// a mean and std no `f32` holds exactly, where the double-precision
    /// operands show. The fused multiply-add read `0xbb808000`, `0x3f119194`
    /// and `0x3f7dfe00` at levels 127, 200 and 254 of the first table.
    #[test]
    fn normalized_levels_carry_the_reference_bits() {
        let half = normalized_levels(&[0.5; 3], &[0.5; 3]);
        for (level, bits) in [
            (0, 0xbf80_0000),
            (1, 0xbf7d_fdfe),
            (2, 0xbf7b_fbfc),
            (3, 0xbf79_f9fa),
            (127, 0xbb80_8080),
            (128, 0x3b80_8100),
            (200, 0x3f11_9192),
            (254, 0x3f7d_fdfe),
            (255, 0x3f80_0000),
        ] {
            for (channel, table) in half.iter().enumerate() {
                assert_eq!(
                    table[level].to_bits(),
                    bits,
                    "level {level} channel {channel}"
                );
            }
        }

        let clip = normalized_levels(
            &[0.48145466, 0.4578275, 0.40821073],
            &[0.26862954, 0.26130258, 0.27577711],
        );
        for (level, bits) in [
            (0, [0xbfe5_68dc, 0xbfe0_44b8, 0xbfbd_77d7]),
            (1, [0xbfe3_8a80, 0xbfde_58f2, 0xbfbb_a5e1]),
            (2, [0xbfe1_ac23, 0xbfdc_6d2c, 0xbfb9_d3ea]),
            (3, [0xbfdf_cdc7, 0xbfda_8165, 0xbfb8_01f4]),
            (127, [0x3d7c_e09e, 0x3e1d_9533, 0x3ea6_c5e5]),
            (128, [0x3d9c_5626, 0x3e2c_f36d, 0x3eae_0dc3]),
            (200, [0x3f90_4f64, 0x3f9f_ee32, 0x3fae_90c9]),
            (254, [0x3ff5_36e5, 0x4003_d502, 0x4008_6d65]),
            (255, [0x3ff7_1542, 0x4004_cae5, 0x4009_5660]),
        ] {
            for (channel, (table, expected)) in clip.iter().zip(bits).enumerate() {
                assert_eq!(
                    table[level].to_bits(),
                    expected,
                    "level {level} channel {channel}"
                );
            }
        }
    }

    /// Every level a byte can take, through the whole image path: the bytes
    /// after the resize, the zero padding and the normalization land on the
    /// reference's bits. The fused multiply-add this replaces differed on 111
    /// of the 256 levels.
    #[test]
    fn every_byte_level_normalizes_as_the_reference_does() {
        let params = MoonVitParams {
            patch_size: 14,
            merge_size: 2,
            in_patch_limit: 65536,
            patch_limit_on_one_side: 512,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        // 256 x 28: every level in every channel, under the patch budget, so
        // the resize is the identity; 24 columns of zero padding follow.
        let level_at = |x: usize, channel: usize| -> u8 {
            let x = x as u8;
            [x, 255 - x, x.wrapping_mul(7)][channel]
        };
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(256, 28, |x, _| {
            Rgb([
                level_at(x as usize, 0),
                level_at(x as usize, 1),
                level_at(x as usize, 2),
            ])
        }));

        let inputs = preprocess(params, &[&image], None).unwrap();

        let (grid_h, grid_w, patch) = (2, 20, 14);
        assert_eq!(
            inputs.encoder_input.shape(),
            &[grid_h * grid_w, 3, patch, patch]
        );
        let got = inputs.encoder_input_flat();
        let mut index = 0;
        for gh in 0..grid_h {
            for gw in 0..grid_w {
                for channel in 0..3 {
                    for py in 0..patch {
                        for px in 0..patch {
                            let x = gw * patch + px;
                            let level = if x < 256 { level_at(x, channel) } else { 0 };
                            let expected = reference_level(level, 0.5, 0.5);
                            assert_eq!(
                                got[index].to_bits(),
                                expected.to_bits(),
                                "patch ({gh}, {gw}) channel {channel} at ({px}, {py}): level {level}"
                            );
                            index += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(index, got.len());
    }

    /// Bytes from the generator the kernel pins share: a 32-bit linear
    /// congruential step per byte, the top byte of the state taken.
    fn generated_bytes(count: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect()
    }

    /// The SHA-256 of the tensor's little-endian f32 bits, in hex.
    fn digest_of(values: &[f32]) -> String {
        use sha2::{Digest, Sha256};

        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        hex(&Sha256::digest(&bytes))
    }

    /// A budget of three patches: a 60x40 image is resized to 36x24 (a ratio
    /// of 5/3 on both axes) and zero padded to 56x28, eight patches.
    fn params_with_a_budget_of_three_patches() -> MoonVitParams {
        MoonVitParams {
            patch_size: 14,
            merge_size: 2,
            in_patch_limit: 3,
            patch_limit_on_one_side: 512,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }

    /// An image above the patch budget is resized with the reference's
    /// kernel. Pillow 12.3.0's `Image.resize(BICUBIC)` of the same 60x40
    /// bytes, zero padded and normalized in the reference's arithmetic, gives
    /// these bits: the digest of the whole tensor and six of its values. The
    /// SIMD Catmull-Rom resize this replaces lands one 8-bit level away on
    /// part of the pixels.
    #[test]
    fn an_image_above_the_patch_budget_is_resized_as_the_reference_resizes_it() {
        let image = DynamicImage::ImageRgb8(
            RgbImage::from_raw(60, 40, generated_bytes(60 * 40 * 3, 7)).unwrap(),
        );

        let inputs = preprocess(params_with_a_budget_of_three_patches(), &[&image], None).unwrap();

        assert_eq!(inputs.encoder_input.shape(), &[8, 3, 14, 14]);
        let got = inputs.encoder_input_flat();
        for (index, bits) in [
            (0, 0xbeda_dada),
            (1, 0x3e0c_8c90),
            (17, 0x3e1c_9ca0),
            (500, 0x3e14_9498),
            (1000, 0x3e9a_9a9c),
            (4703, 0xbf80_0000),
        ] {
            assert_eq!(got[index].to_bits(), bits, "element {index}");
        }
        assert_eq!(
            digest_of(&got[..]),
            "2e7a80e6a16c2eeac525b208f17be6f3f508841d3dd322619ca2d6264b1b2283"
        );
    }

    /// With the background painted after the resize, an alpha-carrying image
    /// is resized premultiplied, as Pillow resizes `RGBA`, and composited at
    /// the new size. The same 60x40 geometry with its first ten columns
    /// transparent, its last ten opaque and the generator's alpha between,
    /// over the 8-pixel chessboard, gives these bits (Pillow 12.3.0 and the
    /// reference's float32 composite). Resampling the four channels without
    /// the premultiplication differs on 1,795 of the 4,704 elements.
    #[test]
    fn an_alpha_image_resized_before_its_background_is_painted_matches_the_reference() {
        use crate::vision::transforms::{TransparentBgConfig, TransparentBgPattern};

        let mut rgba = generated_bytes(60 * 40 * 4, 11);
        for (index, pixel) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            match index % 60 {
                column if column < 10 => pixel[3] = 0,
                column if column >= 50 => pixel[3] = 255,
                _ => {}
            }
        }
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_raw(60, 40, rgba).unwrap());
        let chessboard = TransparentBg {
            config: TransparentBgConfig {
                pattern: TransparentBgPattern::Chessboard,
                chessboard_square_size: 8,
                chessboard_square_on_top_left: true,
                chessboard_white_value: 255,
                chessboard_gray_value: 180,
            },
            stage: TransparentBgFillStage::AfterResize,
        };

        let inputs = preprocess(
            params_with_a_budget_of_three_patches(),
            &[&image],
            Some(chessboard),
        )
        .unwrap();

        assert_eq!(inputs.encoder_input.shape(), &[8, 3, 14, 14]);
        let got = inputs.encoder_input_flat();
        for (index, bits) in [
            (0, 0x3f80_0000),
            (1, 0x3f80_0000),
            (17, 0x3f80_0000),
            (500, 0x3e24_a4a8),
            (1000, 0x3efe_ff00),
            (4703, 0xbf80_0000),
        ] {
            assert_eq!(got[index].to_bits(), bits, "element {index}");
        }
        assert_eq!(
            digest_of(&got[..]),
            "47e02f6a5305085afa077da8ab756e1380052bea0a4433ce584c1f1f7e972a50"
        );
    }
}
