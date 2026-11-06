//! Image transformation functions for vision preprocessing.
//!
//! This module provides composable transforms that match HuggingFace image processor
//! behavior, enabling pure Rust preprocessing without Python dependencies.

use std::{cell::RefCell, f64::consts::PI};

use fast_image_resize::{
    images::{Image as FirImage, ImageRef as FirImageRef},
    IntoImageView, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgb, RgbImage};
use ndarray::{s, Array3, Array4};

use super::{
    execution::{scope as parallel_scope, task_count},
    scratch,
};
pub use crate::error::TransformError;

pub type Result<T> = std::result::Result<T, TransformError>;

/// Extract RGB pixel data from a DynamicImage, avoiding a copy when already RGB8.
/// Returns (width, height, raw_bytes) where raw_bytes is interleaved R,G,B,R,G,B,...
pub fn rgb_bytes(image: &DynamicImage) -> (usize, usize, std::borrow::Cow<'_, [u8]>) {
    match image {
        DynamicImage::ImageRgb8(rgb) => (
            rgb.width() as usize,
            rgb.height() as usize,
            std::borrow::Cow::Borrowed(rgb.as_raw()),
        ),
        _ => {
            let rgb = image.to_rgb8();
            let w = rgb.width() as usize;
            let h = rgb.height() as usize;
            (w, h, std::borrow::Cow::Owned(rgb.into_raw()))
        }
    }
}

/// Background pattern composited underneath a transparent image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransparentBgPattern {
    White,
    Black,
    Gray,
    /// Alternating light/dark squares — the familiar image-editor rendering of
    /// transparency, which some vision models are trained to read as such.
    Chessboard,
}

/// How to flatten an image that carries an alpha channel.
///
/// Field names and defaults mirror the reference `TransparentBgConfig` so a
/// model's `preprocessor_config.json` deserializes straight into this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(default)]
pub struct TransparentBgConfig {
    pub pattern: TransparentBgPattern,
    pub chessboard_square_size: u32,
    pub chessboard_square_on_top_left: bool,
    pub chessboard_white_value: u8,
    pub chessboard_gray_value: u8,
}

impl Default for TransparentBgConfig {
    fn default() -> Self {
        Self {
            pattern: TransparentBgPattern::Black,
            chessboard_square_size: 16,
            chessboard_square_on_top_left: true,
            chessboard_white_value: 255,
            chessboard_gray_value: 200,
        }
    }
}

impl TransparentBgConfig {
    /// Background grey level for pixel `(x, y)`.
    fn background_at(self, x: u32, y: u32) -> u8 {
        match self.pattern {
            TransparentBgPattern::White => 255,
            TransparentBgPattern::Black => 0,
            TransparentBgPattern::Gray => 128,
            TransparentBgPattern::Chessboard => {
                // A square size of 0 would divide by zero here; the reference
                // raises on the same input, so clamp instead of panicking.
                let size = self.chessboard_square_size.max(1);
                let gray_cell = u32::from(self.chessboard_square_on_top_left);
                if (y / size + x / size) % 2 == gray_cell {
                    self.chessboard_gray_value
                } else {
                    self.chessboard_white_value
                }
            }
        }
    }
}

/// Where in the pipeline an alpha-carrying image gets flattened.
///
/// The distinction is load-bearing for [`TransparentBgPattern::Chessboard`]:
/// the board is generated at the resolution of the image it is composited
/// onto, so flattening before vs. after the resize yields different square
/// sizes relative to the content. `BeforeResize` is the default only because
/// that is the reference's fallback when the key is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransparentBgFillStage {
    #[default]
    BeforeResize,
    AfterResize,
}

/// A model's complete alpha-flattening behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TransparentBg {
    pub config: TransparentBgConfig,
    pub stage: TransparentBgFillStage,
}

/// Alpha-composite `image` over the configured background and return RGB.
///
/// Images with no alpha channel convert straight to RGB, matching the
/// reference's early return. Note that dropping alpha (what `to_rgb8` does on
/// its own) is *not* equivalent: a fully transparent pixel usually stores RGB
/// `(0,0,0)`, so it would read as solid black instead of as background.
pub fn fill_transparent_bg(image: &DynamicImage, config: TransparentBgConfig) -> RgbImage {
    if !image.color().has_alpha() {
        return image.to_rgb8();
    }

    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut out = Vec::with_capacity(width as usize * height as usize * 3);

    for (y, row) in rgba.as_raw().chunks_exact(width as usize * 4).enumerate() {
        for (x, px) in row.as_chunks::<4>().0.iter().enumerate() {
            let bg = f32::from(config.background_at(x as u32, y as u32));
            let alpha = f32::from(px[3]) / 255.0;
            let inv = 1.0 - alpha;
            // numpy's `.astype(np.uint8)` truncates rather than rounds, and so
            // does an `as` cast; keep them consistent.
            out.push((alpha * f32::from(px[0]) + inv * bg) as u8);
            out.push((alpha * f32::from(px[1]) + inv * bg) as u8);
            out.push((alpha * f32::from(px[2]) + inv * bg) as u8);
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "out holds exactly width*height*3 bytes by construction"
    )]
    RgbImage::from_raw(width, height, out).expect("buffer matches image dimensions")
}

/// Deinterleave interleaved RGB bytes into separate R, G, B f32 planes with
/// per-channel `scale` and `bias`: `plane[c][i] = rgb[i*3 + c] * scale[c] + bias[c]`.
///
/// Processes 8 pixels at a time so the compiler can unroll and auto-vectorize
/// the stride-3 gather pattern.
pub fn deinterleave_rgb_to_planes(
    rgb: &[u8],
    r_plane: &mut [f32],
    g_plane: &mut [f32],
    b_plane: &mut [f32],
    scale: [f32; 3],
    bias: [f32; 3],
) {
    deinterleave_banded(rgb, r_plane, g_plane, b_plane, move |rgb, r, g, b| {
        deinterleave_contiguous(rgb, r, g, b, scale, bias);
    });
}

/// [`deinterleave_rgb_to_planes`] through a per-channel table instead of a
/// scale and a bias: `plane[c][i] = table[c][rgb[i*3 + c]]`, for a reference
/// whose arithmetic no single multiply-add reproduces in the last bit.
fn deinterleave_rgb_to_planes_by_table(
    rgb: &[u8],
    r_plane: &mut [f32],
    g_plane: &mut [f32],
    b_plane: &mut [f32],
    table: &[[f32; 256]; 3],
) {
    deinterleave_banded(rgb, r_plane, g_plane, b_plane, move |rgb, r, g, b| {
        deinterleave_contiguous_by_table(rgb, r, g, b, table);
    });
}

/// Band the pixels across the pool's threads and run `kernel` over each
/// band's bytes and plane slices. Each output element depends only on its
/// own input byte, so banding is BIT-IDENTICAL (elementwise f32, no
/// reduction). Small images stay serial.
fn deinterleave_banded<K>(
    rgb: &[u8],
    r_plane: &mut [f32],
    g_plane: &mut [f32],
    b_plane: &mut [f32],
    kernel: K,
) where
    K: Fn(&[u8], &mut [f32], &mut [f32], &mut [f32]) + Copy + Send + Sync,
{
    let pixels = r_plane.len();
    debug_assert_eq!(pixels, g_plane.len());
    debug_assert_eq!(pixels, b_plane.len());
    debug_assert!(rgb.len() >= pixels * 3);

    let nthreads = par_threads(pixels * 3 * 4, pixels);
    if nthreads <= 1 {
        kernel(rgb, r_plane, g_plane, b_plane);
        return;
    }
    let chunk = pixels.div_ceil(nthreads);
    let (mut rr, mut gg, mut bb) = (r_plane, g_plane, b_plane);
    parallel_scope(|s| {
        let mut p0 = 0usize;
        while p0 < pixels {
            let n = chunk.min(pixels - p0);
            let (rb, rt) = rr.split_at_mut(n);
            let (gb, gt) = gg.split_at_mut(n);
            let (bbnd, bt) = bb.split_at_mut(n);
            rr = rt;
            gg = gt;
            bb = bt;
            let rgb_band = &rgb[p0 * 3..(p0 + n) * 3];
            s.spawn(move |_| kernel(rgb_band, rb, gb, bbnd));
            p0 += n;
        }
    });
}

/// Deinterleave a contiguous pixel range (planes/rgb already sliced to the band).
fn deinterleave_contiguous(
    rgb: &[u8],
    r_plane: &mut [f32],
    g_plane: &mut [f32],
    b_plane: &mut [f32],
    scale: [f32; 3],
    bias: [f32; 3],
) {
    let pixels = r_plane.len();
    let full_blocks = pixels / 8;
    let remainder = pixels % 8;

    for block in 0..full_blocks {
        let dst = block * 8;
        let src_base = dst * 3;
        let src = &rgb[src_base..src_base + 24];
        let rd = &mut r_plane[dst..dst + 8];
        let gd = &mut g_plane[dst..dst + 8];
        let bd = &mut b_plane[dst..dst + 8];

        for i in 0..8 {
            let s = i * 3;
            rd[i] = src[s] as f32 * scale[0] + bias[0];
            gd[i] = src[s + 1] as f32 * scale[1] + bias[1];
            bd[i] = src[s + 2] as f32 * scale[2] + bias[2];
        }
    }

    let tail_dst = full_blocks * 8;
    let tail_src = tail_dst * 3;
    for i in 0..remainder {
        let s = tail_src + i * 3;
        r_plane[tail_dst + i] = rgb[s] as f32 * scale[0] + bias[0];
        g_plane[tail_dst + i] = rgb[s + 1] as f32 * scale[1] + bias[1];
        b_plane[tail_dst + i] = rgb[s + 2] as f32 * scale[2] + bias[2];
    }
}

/// [`deinterleave_contiguous`] through a per-channel table: one load per byte.
fn deinterleave_contiguous_by_table(
    rgb: &[u8],
    r_plane: &mut [f32],
    g_plane: &mut [f32],
    b_plane: &mut [f32],
    table: &[[f32; 256]; 3],
) {
    let [red, green, blue] = table;
    let (pixels, _) = rgb.as_chunks::<3>();
    for (((pixel, r), g), b) in pixels.iter().zip(r_plane).zip(g_plane).zip(b_plane) {
        *r = red[usize::from(pixel[0])];
        *g = green[usize::from(pixel[1])];
        *b = blue[usize::from(pixel[2])];
    }
}

/// Build a [C, H, W] f32 tensor from interleaved RGB bytes with per-channel
/// `scale` and `bias`: `output[c][i] = raw[i*3 + c] * scale[c] + bias[c]`.
fn build_planar_tensor(
    raw: &[u8],
    w: usize,
    h: usize,
    scale: [f32; 3],
    bias: [f32; 3],
) -> Array3<f32> {
    let pixels = h * w;
    // Pooled: this large per-image buffer is the data plane's hottest allocation.
    let mut data = scratch::take_f32(3 * pixels);
    let (r_plane, rest) = data.split_at_mut(pixels);
    let (g_plane, b_plane) = rest.split_at_mut(pixels);

    deinterleave_rgb_to_planes(raw, r_plane, g_plane, b_plane, scale, bias);

    #[expect(
        clippy::expect_used,
        reason = "data has exactly 3*h*w elements by construction"
    )]
    Array3::from_shape_vec((3, h, w), data).expect("shape matches pre-allocated buffer")
}

/// Paint an `image_w` x `image_h` image, given as interleaved RGB bytes, onto
/// the top-left of a `canvas_w` x `canvas_h` planar canvas: `planes` holds the
/// R, G and B planes back to back (`3 * canvas_w * canvas_h` elements), every
/// painted byte maps through `byte * scale[c] + bias[c]`, and the canvas
/// outside the image holds the mapped black, `bias[c]`. An image larger than
/// the canvas is cropped at its right and bottom edges.
pub fn paint_rgb_onto_planes(
    rgb: &[u8],
    image: (usize, usize),
    canvas: (usize, usize),
    scale: [f32; 3],
    bias: [f32; 3],
    planes: &mut [f32],
) {
    paint_rows(rgb, image, canvas, bias, planes, |row, r, g, b| {
        deinterleave_rgb_to_planes(row, r, g, b, scale, bias);
    });
}

/// [`paint_rgb_onto_planes`] through a per-channel table: every painted byte
/// maps through `table[c][byte]` and the canvas outside the image holds
/// `table[c][0]`, what a zero byte of padding maps to under the same
/// arithmetic.
pub(crate) fn paint_rgb_onto_planes_by_table(
    rgb: &[u8],
    image: (usize, usize),
    canvas: (usize, usize),
    table: &[[f32; 256]; 3],
    planes: &mut [f32],
) {
    let black = [table[0][0], table[1][0], table[2][0]];
    paint_rows(rgb, image, canvas, black, planes, |row, r, g, b| {
        deinterleave_rgb_to_planes_by_table(row, r, g, b, table);
    });
}

/// The canvas fill and the row walk of the paints: `black` fills the canvas,
/// then `paint_row` maps each image row's bytes onto its three plane slices.
fn paint_rows<K>(
    rgb: &[u8],
    (image_w, image_h): (usize, usize),
    (canvas_w, canvas_h): (usize, usize),
    black: [f32; 3],
    planes: &mut [f32],
    paint_row: K,
) where
    K: Fn(&[u8], &mut [f32], &mut [f32], &mut [f32]),
{
    let canvas_pixels = canvas_w * canvas_h;
    debug_assert_eq!(planes.len(), 3 * canvas_pixels);
    debug_assert!(rgb.len() >= image_w * image_h * 3);
    let (r_plane, rest) = planes.split_at_mut(canvas_pixels);
    let (g_plane, b_plane) = rest.split_at_mut(canvas_pixels);

    r_plane.fill(black[0]);
    g_plane.fill(black[1]);
    b_plane.fill(black[2]);

    let painted_w = image_w.min(canvas_w);
    let painted_h = image_h.min(canvas_h);
    for y in 0..painted_h {
        let src_row = &rgb[y * image_w * 3..(y * image_w + painted_w) * 3];
        let dst = y * canvas_w;
        paint_row(
            src_row,
            &mut r_plane[dst..dst + painted_w],
            &mut g_plane[dst..dst + painted_w],
            &mut b_plane[dst..dst + painted_w],
        );
    }
}

/// Convert image to tensor [C, H, W] normalized to [0, 1].
///
/// This matches the default behavior of `torchvision.transforms.ToTensor()`.
pub fn to_tensor(image: &DynamicImage) -> Array3<f32> {
    let (w, h, raw) = rgb_bytes(image);
    let s = 1.0 / 255.0;
    build_planar_tensor(&raw, w, h, [s, s, s], [0.0, 0.0, 0.0])
}

/// Convert image to tensor [C, H, W] without normalization (keeps [0, 255]).
#[cfg(test)]
pub fn to_tensor_no_norm(image: &DynamicImage) -> Array3<f32> {
    let (w, h, raw) = rgb_bytes(image);
    build_planar_tensor(&raw, w, h, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
}

/// Normalize tensor per channel: (x - mean) / std.
///
/// This matches `torchvision.transforms.Normalize(mean, std)`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [C, H, W]
/// * `mean` - Per-channel mean values
/// * `std` - Per-channel standard deviation values
pub fn normalize(tensor: &mut Array3<f32>, mean: &[f64; 3], std: &[f64; 3]) {
    let [h, w] = [tensor.shape()[1], tensor.shape()[2]];
    let pixels = h * w;

    if let Some(flat) = tensor.as_slice_mut() {
        // Fast path: contiguous memory, process channel planes directly
        for c in 0..3 {
            let mean_c = mean[c] as f32;
            let inv_std_c = 1.0 / std[c] as f32;
            let plane = &mut flat[c * pixels..(c + 1) * pixels];
            for v in plane.iter_mut() {
                *v = (*v - mean_c) * inv_std_c;
            }
        }
    } else {
        for c in 0..3 {
            let mean_c = mean[c] as f32;
            let std_c = std[c] as f32;
            tensor
                .slice_mut(s![c, .., ..])
                .mapv_inplace(|v| (v - mean_c) / std_c);
        }
    }
}

/// Convert image to tensor and normalize in a single pass.
///
/// Fuses `to_tensor` (u8→f32 with /255) and `normalize` ((x-mean)/std)
/// into one loop to avoid an extra pass over the data.
pub fn to_tensor_and_normalize(
    image: &DynamicImage,
    mean: &[f64; 3],
    std: &[f64; 3],
) -> Array3<f32> {
    let (w, h, raw) = rgb_bytes(image);
    // Fused: (pixel/255 - mean) / std = pixel * (1/(255*std)) - mean/std
    let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c] as f32));
    let bias: [f32; 3] = std::array::from_fn(|c| -(mean[c] as f32) / (std[c] as f32));
    build_planar_tensor(&raw, w, h, scale, bias)
}

/// Rescale tensor by a constant factor.
///
/// Used when `do_rescale=True` in HuggingFace configs (typically 1/255).
pub fn rescale(tensor: &mut Array3<f32>, factor: f64) {
    let factor = factor as f32;
    tensor.mapv_inplace(|v| v * factor);
}

/// Python-compatible rounding (banker's rounding / round half to even).
/// Matches Python's `round()` where 0.5 rounds to the nearest even number
/// (`12.5 -> 12`, `13.5 -> 14`), unlike Rust's `f64::round()` which rounds
/// half away from zero.
#[inline]
pub fn round_half_to_even(x: f64) -> f64 {
    let rounded = x.round();
    // Check if we're exactly at a .5 case
    if (x - x.floor() - 0.5).abs() < 1e-9 {
        // Round to nearest even
        if rounded as i64 % 2 != 0 {
            return rounded - 1.0;
        }
    }
    rounded
}

/// Map `image` crate filter types to `fast_image_resize` algorithm.
fn to_fir_algorithm(filter: FilterType) -> ResizeAlg {
    use fast_image_resize::FilterType as FirFilter;
    match filter {
        FilterType::Nearest => ResizeAlg::Nearest,
        FilterType::Triangle => ResizeAlg::Convolution(FirFilter::Bilinear),
        FilterType::CatmullRom => ResizeAlg::Convolution(FirFilter::CatmullRom),
        FilterType::Gaussian => ResizeAlg::Convolution(FirFilter::Gaussian),
        FilterType::Lanczos3 => ResizeAlg::Convolution(FirFilter::Lanczos3),
    }
}

thread_local! {
    static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
}

/// Resize image to exact dimensions using SIMD-accelerated resizer.
///
/// # Arguments
/// * `image` - Input image
/// * `width` - Target width
/// * `height` - Target height
/// * `filter` - Interpolation filter (Nearest, Triangle/Bilinear, CatmullRom/Bicubic, Lanczos3)
///
/// Alpha-carrying input is premultiplied for the convolution and un-multiplied
/// afterwards (`fast_image_resize` defaults `mul_div_alpha` to `true`), which is
/// what `PIL.Image.resize` does for `RGBA`.
pub fn resize(image: &DynamicImage, width: u32, height: u32, filter: FilterType) -> DynamicImage {
    let pixel_type = match image.pixel_type() {
        Some(pt) => pt,
        None => return image.resize_exact(width, height, filter),
    };
    let mut dst = FirImage::new(width, height, pixel_type);
    let options = ResizeOptions::new().resize_alg(to_fir_algorithm(filter));
    let ok = RESIZER.with(|r| r.borrow_mut().resize(image, &mut dst, &options).is_ok());
    if !ok {
        return image.resize_exact(width, height, filter);
    }
    fir_image_to_dynamic(dst, width, height, image, filter)
}

// Match image 0.25's f32 Catmull-Rom weights, normalization, and sample order.
fn catmull_rom_weights(input: u32, output: u32) -> Vec<(usize, Vec<f32>)> {
    let ratio = input as f32 / output as f32;
    let scale = ratio.max(1.0);
    (0..output)
        .map(|i| {
            let center = (i as f32 + 0.5) * ratio;
            let left = ((center - 2.0 * scale).floor() as i64).clamp(0, input as i64 - 1) as usize;
            let right = ((center + 2.0 * scale).ceil() as i64).clamp(left as i64 + 1, input as i64)
                as usize;
            let mut weights: Vec<_> = (left..right)
                .map(|j| {
                    let a = ((j as f32 - (center - 0.5)) / scale).abs();
                    (if a < 1.0 {
                        9.0 * a.powi(3) - 15.0 * a.powi(2) + 6.0
                    } else if a < 2.0 {
                        -3.0 * a.powi(3) + 15.0 * a.powi(2) - 24.0 * a + 12.0
                    } else {
                        0.0
                    }) / 6.0
                })
                .collect();
            let sum: f32 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            (left, weights)
        })
        .collect()
}

/// Match `image::imageops::resize(..., CatmullRom)` byte-for-byte for RGB8.
/// Fuse both separable passes per output row: the temporary row stays in cache
/// instead of materializing an RGBA32F image and sampling it column-first.
/// Run on the caller: lazy decode holds an ImageFrame initialization cell,
/// and pool workers may already be waiting for that cell to become ready.
pub(crate) fn resize_catmull_rom_rgb(src: &RgbImage, width: u32, height: u32) -> RgbImage {
    if src.width() == 0 || src.height() == 0 || width == 0 || height == 0 {
        return image::imageops::resize(src, width, height, FilterType::CatmullRom);
    }
    let xs = catmull_rom_weights(src.width(), width);
    let ys = catmull_rom_weights(src.height(), height);
    let src_stride = src.width() as usize * 3;
    let dst_stride = width as usize * 3;
    let mut dst = RgbImage::new(width, height);
    if dst.set_color_space(src.color_space()).is_err() {
        return image::imageops::resize(src, width, height, FilterType::CatmullRom);
    }
    let mut row = vec![0.0f32; src_stride];
    for (index, dst_row) in dst.as_mut().chunks_mut(dst_stride).enumerate() {
        row.fill(0.0);
        let (top, weights) = &ys[index];
        for (dy, w) in weights.iter().enumerate() {
            let start = (top + dy) * src_stride;
            for (out, &value) in row.iter_mut().zip(&src.as_raw()[start..start + src_stride]) {
                *out += value as f32 * w;
            }
        }
        for (pixel, (left, weights)) in dst_row.as_chunks_mut::<3>().0.iter_mut().zip(&xs) {
            let mut sums = [0.0f32; 3];
            for (dx, w) in weights.iter().enumerate() {
                for c in 0..3 {
                    sums[c] += row[(left + dx) * 3 + c] * w;
                }
            }
            for c in 0..3 {
                pixel[c] = sums[c].clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    dst
}

/// Resize borrowed interleaved RGB bytes without first materializing an
/// `image::RgbImage` over an owned input buffer.
pub fn resize_rgb_bytes(
    data: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> Result<RgbImage> {
    let src = FirImageRef::new(width, height, data, PixelType::U8x3)
        .map_err(|e| TransformError::ShapeError(format!("invalid RGB source image: {e}")))?;
    let mut dst = FirImage::new(target_width, target_height, PixelType::U8x3);
    let options = ResizeOptions::new().resize_alg(to_fir_algorithm(filter));
    RESIZER
        .with(|r| r.borrow_mut().resize(&src, &mut dst, &options))
        .map_err(|e| TransformError::ShapeError(format!("RGB resize failed: {e}")))?;

    RgbImage::from_raw(target_width, target_height, dst.into_vec()).ok_or_else(|| {
        TransformError::ShapeError(format!(
            "failed to build resized RGB image for {target_width}x{target_height}"
        ))
    })
}

/// Convert a `fast_image_resize::Image` back to a `DynamicImage`.
///
/// Falls back to the `image` crate resize for unhandled pixel formats.
fn fir_image_to_dynamic(
    img: FirImage<'_>,
    width: u32,
    height: u32,
    source: &DynamicImage,
    filter: FilterType,
) -> DynamicImage {
    let buf = img.into_vec();
    match source {
        DynamicImage::ImageRgb8(_) => {
            RgbImage::from_raw(width, height, buf).map(DynamicImage::ImageRgb8)
        }
        DynamicImage::ImageRgba8(_) => {
            image::RgbaImage::from_raw(width, height, buf).map(DynamicImage::ImageRgba8)
        }
        DynamicImage::ImageLuma8(_) => {
            image::GrayImage::from_raw(width, height, buf).map(DynamicImage::ImageLuma8)
        }
        _ => None,
    }
    .unwrap_or_else(|| source.resize_exact(width, height, filter))
}

// ---------------------------------------------------------------------------
// Pillow-exact bicubic resize.
//
// The image processors whose reference resizes via `PIL.Image.resize(size,
// BICUBIC)` on the uint8 image take this kernel; the ones whose reference is
// torchvision-backed (the Qwen vision families, Gemma 4) take the `uint8`
// resampler below, which differs only in how the taps are quantized. The SIMD
// `fast_image_resize` path above is the same filter *family* (Catmull-Rom,
// a=-0.5) but diverges bit-wise on non-integer ratios (support scaling +
// fixed-point details), which the vision encoder amplifies into a large
// embedding shift. This routine replicates Pillow's `Resample.c` algorithm
// exactly, validated against Pillow.
const PIL_PRECISION_BITS: u32 = 32 - 8 - 2;
const PIL_BICUBIC_SUPPORT: f64 = 2.0;
const PIL_LANCZOS_SUPPORT: f64 = 3.0;

#[derive(Clone, Copy)]
pub(super) enum PilResizeFilter {
    Bicubic,
    Lanczos,
}

impl PilResizeFilter {
    fn support(self) -> f64 {
        match self {
            Self::Bicubic => PIL_BICUBIC_SUPPORT,
            Self::Lanczos => PIL_LANCZOS_SUPPORT,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Self::Bicubic => pil_cubic(x),
            Self::Lanczos => pil_lanczos(x),
        }
    }
}

#[inline]
fn pil_cubic(x: f64) -> f64 {
    // Keys cubic with a = -0.5 (Pillow's BICUBIC).
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

#[inline]
fn pil_sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x = x * PI;
        x.sin() / x
    }
}

#[inline]
fn pil_lanczos(x: f64) -> f64 {
    let x = x.abs();
    if x < PIL_LANCZOS_SUPPORT {
        pil_sinc(x) * pil_sinc(x / PIL_LANCZOS_SUPPORT)
    } else {
        0.0
    }
}

/// Pillow `precompute_coeffs` for one axis: the normalized floating-point
/// kernel per output index with its source bounds `(start, count)`.
fn pil_normalized_weights(
    in_size: usize,
    out_size: usize,
    filter: PilResizeFilter,
) -> (Vec<(usize, usize)>, Vec<Vec<f64>>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale >= 1.0 { scale } else { 1.0 };
    let support = filter.support() * filterscale;
    let inv = 1.0 / filterscale;

    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let xmin = xmin as usize;
        let xmax = (xmax as usize).saturating_sub(xmin);

        let mut w = vec![0.0_f64; xmax];
        let mut tot = 0.0;
        for (x, wx) in w.iter_mut().enumerate() {
            let v = filter.weight(((x + xmin) as f64 - center + 0.5) * inv);
            *wx = v;
            tot += v;
        }
        if tot != 0.0 {
            for wx in &mut w {
                *wx /= tot;
            }
        }
        bounds.push((xmin, xmax));
        weights.push(w);
    }
    (bounds, weights)
}

/// How a normalized kernel is turned into integer taps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KernelQuantization {
    /// Pillow's `normalize_coeffs_8bpc`: a fixed 22-bit scale.
    Pillow,
    /// PyTorch's `uint8` resampler (`interpolate(.., antialias=True)` on a
    /// `uint8` tensor, which torchvision's `resize` and so transformers'
    /// torchvision-backed image processors run): 16-bit taps at the finest
    /// scale that keeps the largest weight below 2^15.
    TorchUint8,
}

/// One axis of a separable resample in fixed point: per output index the
/// source bounds `(start, count)` and the integer taps, plus the shift that
/// undoes their scale.
///
/// The taps and the sums they produce fit 32 bits under either quantization,
/// which is how Pillow's own resampler (`INT32` coefficients and
/// accumulators, the two spare bits of its 22-bit precision covering the
/// negative lobes) and the torch `uint8` one compute: a tap is below 2^23 in
/// magnitude, a pixel below 2^8, and the taps of one output sum to one in
/// absolute value within a factor of two, so `half + Σ pixel · tap` stays
/// below 2^31.
pub(super) struct FixedKernel {
    pub(super) bounds: Vec<(usize, usize)>,
    pub(super) taps: Vec<Vec<i32>>,
    pub(super) precision: u32,
}

impl FixedKernel {
    pub(super) fn new(
        in_size: usize,
        out_size: usize,
        filter: PilResizeFilter,
        quantization: KernelQuantization,
    ) -> Self {
        let (bounds, weights) = pil_normalized_weights(in_size, out_size, filter);
        let precision = match quantization {
            KernelQuantization::Pillow => PIL_PRECISION_BITS,
            KernelQuantization::TorchUint8 => {
                // PyTorch's `_compute_index_ranges_int16_weights`: the largest
                // weight is the (positive) centre tap of some output index.
                let max_weight = weights
                    .iter()
                    .flatten()
                    .fold(0.0_f64, |max, &weight| max.max(weight));
                let mut precision = 0;
                while precision < PIL_PRECISION_BITS {
                    let next = (0.5 + max_weight * f64::from(1_u32 << (precision + 1))) as i64;
                    if next >= 1 << 15 {
                        break;
                    }
                    precision += 1;
                }
                precision
            }
        };
        let coeff_scale = f64::from(1_u32 << precision);
        // Round half away from zero into fixed point (Pillow and PyTorch alike).
        let taps = weights
            .iter()
            .map(|w| {
                w.iter()
                    .map(|&c| {
                        if c < 0.0 {
                            (-0.5 + c * coeff_scale) as i32
                        } else {
                            (0.5 + c * coeff_scale) as i32
                        }
                    })
                    .collect()
            })
            .collect();
        Self {
            bounds,
            taps,
            precision,
        }
    }

    /// The rounding term an accumulator starts from.
    #[inline]
    fn half(&self) -> i32 {
        1_i32 << (self.precision - 1)
    }

    /// `(sum >> precision)` clamped to a byte: Pillow's `clip8`.
    #[inline]
    fn clip8(&self, sum: i32) -> u8 {
        (sum >> self.precision).clamp(0, 255) as u8
    }
}

/// The computation the fixed-point passes replace, for the tests: per
/// sample, the kernel's own taps summed in `i64`, width first then height,
/// `(sum >> precision)` clamped to a byte; `C` interleaved channels.
#[cfg(test)]
pub(super) fn per_sample_reference_channels<const C: usize>(
    data: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    filter: PilResizeFilter,
    quantization: KernelQuantization,
) -> Vec<u8> {
    fn clip(kernel: &FixedKernel, v: i64) -> u8 {
        (v >> kernel.precision).clamp(0, 255) as u8
    }
    let mut mid = data.to_vec();
    if in_w != out_w {
        let k = FixedKernel::new(in_w, out_w, filter, quantization);
        let half = 1_i64 << (k.precision - 1);
        let mut out = vec![0u8; in_h * out_w * C];
        for y in 0..in_h {
            for xx in 0..out_w {
                let (x0, n) = k.bounds[xx];
                for c in 0..C {
                    let mut s = half;
                    for x in 0..n {
                        s +=
                            i64::from(data[(y * in_w + x0 + x) * C + c]) * i64::from(k.taps[xx][x]);
                    }
                    out[(y * out_w + xx) * C + c] = clip(&k, s);
                }
            }
        }
        mid = out;
    }
    if in_h == out_h {
        return mid;
    }
    let k = FixedKernel::new(in_h, out_h, filter, quantization);
    let half = 1_i64 << (k.precision - 1);
    let mut out = vec![0u8; out_h * out_w * C];
    for yy in 0..out_h {
        let (y0, n) = k.bounds[yy];
        for x in 0..out_w * C {
            let mut s = half;
            for y in 0..n {
                s += i64::from(mid[(y0 + y) * out_w * C + x]) * i64::from(k.taps[yy][y]);
            }
            out[yy * out_w * C + x] = clip(&k, s);
        }
    }
    out
}

/// [`per_sample_reference_channels`] for RGB.
#[cfg(test)]
pub(super) fn per_sample_reference(
    data: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    filter: PilResizeFilter,
    quantization: KernelQuantization,
) -> Vec<u8> {
    per_sample_reference_channels::<3>(data, in_w, in_h, out_w, out_h, filter, quantization)
}

/// Number of threads to split an elementwise or row-banded preprocessing pass
/// across. Each output row/element is independent, so banding work over threads
/// yields BIT-IDENTICAL output: no shared accumulation and no inner-loop order
/// changes. Small images run serial to avoid thread-spawn overhead.
pub(crate) fn par_threads(out_bytes: usize, out_rows: usize) -> usize {
    task_count(out_bytes, out_rows, 32)
}

/// Output rows `[y0, y0 + band.len() / (out_w * C))` of the width pass, the
/// portable arm of [`WidthPass`]: each output pixel sums its taps over the
/// source pixels of its row, the `C` channels side by side, in the 32-bit
/// arithmetic of [`FixedKernel`].
fn pil_h_band<const C: usize>(
    src: &[u8],
    kernel: &FixedKernel,
    in_w: usize,
    out_w: usize,
    y0: usize,
    band: &mut [u8],
) {
    let row_in = in_w * C;
    for (i, out_row) in band.chunks_exact_mut(out_w * C).enumerate() {
        let row = &src[(y0 + i) * row_in..(y0 + i + 1) * row_in];
        pil_h_outputs::<C>(row, kernel, 0, out_row);
    }
}

/// Outputs `first..first + out.len() / C` of one row of the width pass, one
/// at a time: output `x` sums its taps over source pixels `x0..x0 + n`.
fn pil_h_outputs<const C: usize>(row: &[u8], kernel: &FixedKernel, first: usize, out: &mut [u8]) {
    let half = kernel.half();
    for ((out, &(x0, n)), taps) in out
        .as_chunks_mut::<C>()
        .0
        .iter_mut()
        .zip(&kernel.bounds[first..])
        .zip(&kernel.taps[first..])
    {
        let mut sums = [half; C];
        for (pixel, &tap) in row[x0 * C..(x0 + n) * C]
            .as_chunks::<C>()
            .0
            .iter()
            .zip(taps)
        {
            for (sum, &value) in sums.iter_mut().zip(pixel) {
                *sum += i32::from(value) * tap;
            }
        }
        *out = sums.map(|sum| kernel.clip8(sum));
    }
}

/// How [`pil_resample_horizontal`] runs its rows: the portable per-output
/// loop, or, on an x86-64 processor that reports AVX2, the eight-tap vector
/// steps of [`avx2::h_band`] over the kernel's taps padded for them.
enum WidthPass {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2(PaddedTaps),
}

impl WidthPass {
    #[cfg(target_arch = "x86_64")]
    fn new(kernel: &FixedKernel, in_w: usize) -> Self {
        if std::arch::is_x86_feature_detected!("avx2") {
            Self::Avx2(PaddedTaps::new(kernel, in_w))
        } else {
            Self::Portable
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn new(_kernel: &FixedKernel, _in_w: usize) -> Self {
        Self::Portable
    }

    /// Output rows `[y0, y0 + band.len() / (out_w * C))` of the width pass;
    /// the vector arm detects its feature and calls in one place, as the
    /// crate's NEON kernels and the FMA pass keep them.
    #[cfg_attr(
        target_arch = "x86_64",
        expect(
            unsafe_code,
            reason = "the target-feature function runs only after this function detected its feature"
        )
    )]
    fn rows<const C: usize>(
        &self,
        src: &[u8],
        kernel: &FixedKernel,
        in_w: usize,
        out_w: usize,
        y0: usize,
        band: &mut [u8],
    ) {
        match self {
            Self::Portable => pil_h_band::<C>(src, kernel, in_w, out_w, y0, band),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(padded) => {
                if !std::arch::is_x86_feature_detected!("avx2") {
                    return pil_h_band::<C>(src, kernel, in_w, out_w, y0, band);
                }
                // SAFETY: `is_x86_feature_detected!("avx2")` answered true just
                // above, the only requirement of a `target_feature(enable =
                // "avx2")` function.
                unsafe { avx2::h_band::<C>(src, kernel, padded, in_w, out_w, y0, band) }
            }
        }
    }
}

/// The width pass's taps as [`avx2::h_band`] reads them: every output's
/// taps zero-padded to `stride`, the longest window rounded up to eight, at
/// `taps[x * stride..]`; and `vector_outputs`, how many leading outputs keep
/// their padded window `x0..x0 + stride` inside the row. Windows start
/// monotonically along the axis, so the outputs past that count are a
/// suffix at the right edge, which take the per-output loop.
#[cfg(target_arch = "x86_64")]
struct PaddedTaps {
    taps: Vec<i32>,
    stride: usize,
    vector_outputs: usize,
}

#[cfg(target_arch = "x86_64")]
impl PaddedTaps {
    /// Taps per vector step: one 32-bit lane each.
    const STEP: usize = 8;

    fn new(kernel: &FixedKernel, in_w: usize) -> Self {
        let longest = kernel.bounds.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let stride = longest.max(1).next_multiple_of(Self::STEP);
        let mut taps = vec![0_i32; kernel.bounds.len() * stride];
        for (padded, output_taps) in taps.chunks_exact_mut(stride).zip(&kernel.taps) {
            padded[..output_taps.len()].copy_from_slice(output_taps);
        }
        let vector_outputs = kernel
            .bounds
            .iter()
            .position(|&(x0, _)| x0 + stride > in_w)
            .unwrap_or(kernel.bounds.len());
        Self {
            taps,
            stride,
            vector_outputs,
        }
    }
}

/// Output rows `[y0, y0 + band.len() / row_len)` of the height pass: each
/// output row sums its taps over whole source rows, one tap's row at a time,
/// so the inner loop runs along the row. Inlined into its callers so that
/// the loop compiles to the caller's instruction set ([`pil_v_rows`]'s AVX2
/// arm).
#[inline(always)]
fn pil_v_band(src: &[u8], kernel: &FixedKernel, row_len: usize, y0: usize, band: &mut [u8]) {
    let half = kernel.half();
    let mut sums = vec![0_i32; row_len];
    for (i, out_row) in band.chunks_exact_mut(row_len).enumerate() {
        let (start, n) = kernel.bounds[y0 + i];
        sums.fill(half);
        for (k, &tap) in kernel.taps[y0 + i].iter().take(n).enumerate() {
            let row = &src[(start + k) * row_len..(start + k + 1) * row_len];
            for (sum, &value) in sums.iter_mut().zip(row) {
                *sum += i32::from(value) * tap;
            }
        }
        for (out, &sum) in out_row.iter_mut().zip(&sums) {
            *out = kernel.clip8(sum);
        }
    }
}

/// [`pil_v_band`] on the AVX2 unit when the processor has one, the portable
/// loop otherwise: the detection and the call sit in one function, as the
/// crate's NEON kernels and the FMA pass keep them.
#[cfg(target_arch = "x86_64")]
#[expect(
    unsafe_code,
    reason = "the target-feature function runs only after this function detected its feature"
)]
fn pil_v_rows(src: &[u8], kernel: &FixedKernel, row_len: usize, y0: usize, band: &mut [u8]) {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return pil_v_band(src, kernel, row_len, y0, band);
    }
    // SAFETY: `is_x86_feature_detected!("avx2")` answered true just above, the
    // only requirement of a `target_feature(enable = "avx2")` function.
    unsafe { avx2::v_band(src, kernel, row_len, y0, band) }
}

#[cfg(not(target_arch = "x86_64"))]
fn pil_v_rows(src: &[u8], kernel: &FixedKernel, row_len: usize, y0: usize, band: &mut [u8]) {
    pil_v_band(src, kernel, row_len, y0, band);
}

/// The Pillow-exact passes compiled for AVX2, the vector arms of
/// [`WidthPass`] and [`pil_v_rows`]: the same 32-bit integer arithmetic as
/// the portable passes, eight lanes at a time. Integer sums are the same in
/// any order, so the bytes are those of the portable passes and of Pillow.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_blend_epi32, _mm256_castsi256_si128,
        _mm256_cvtepu8_epi32, _mm256_extracti128_si256, _mm256_mullo_epi32,
        _mm256_permute2x128_si256, _mm256_permutevar8x32_epi32, _mm256_setr_epi32,
        _mm256_setzero_si256, _mm256_zextsi128_si256, _mm_add_epi32, _mm_cvtsi128_si32,
        _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi32,
        _mm_setzero_si128, _mm_sra_epi32,
    };

    use super::{pil_h_outputs, pil_v_band, FixedKernel, PaddedTaps};

    /// Lane `l` of the `q`th eight-byte group of a chunk of eight
    /// `C`-channel source pixels holds pixel `(8q + l) / C`, so multiplies
    /// that pixel's tap: the lane indices that spread the chunk's eight taps
    /// over the group.
    const fn tap_lanes<const C: usize>(q: usize) -> [i32; 8] {
        let mut lanes = [0_i32; 8];
        let mut l = 0;
        while l < 8 {
            lanes[l] = ((8 * q + l) / C) as i32;
            l += 1;
        }
        lanes
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn lanes(values: [i32; 8]) -> __m256i {
        _mm256_setr_epi32(
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        )
    }

    /// Output rows `[y0, y0 + band.len() / (out_w * C))` of the width pass.
    /// Per output, the padded taps go eight at a time against the eight
    /// source pixels they weigh: each eight-byte group of those pixels' `8C`
    /// bytes multiplies, by the taps spread to its lanes, into an accumulator
    /// of its own, `C` of them, and `channel_sums` gathers the channel totals
    /// at the end. The outputs whose padded window would cross the row's end
    /// take the per-output loop.
    #[target_feature(enable = "avx2")]
    pub(super) fn h_band<const C: usize>(
        src: &[u8],
        kernel: &FixedKernel,
        padded: &PaddedTaps,
        in_w: usize,
        out_w: usize,
        y0: usize,
        band: &mut [u8],
    ) {
        let row_in = in_w * C;
        let stride = padded.stride;
        let spreads: [__m256i; C] = std::array::from_fn(|q| lanes(tap_lanes::<C>(q)));
        for (i, out_row) in band.chunks_exact_mut(out_w * C).enumerate() {
            let row = &src[(y0 + i) * row_in..(y0 + i + 1) * row_in];
            let (vector, rest) = out_row.split_at_mut(padded.vector_outputs * C);
            for ((out, &(x0, _)), taps) in vector
                .as_chunks_mut::<C>()
                .0
                .iter_mut()
                .zip(&kernel.bounds)
                .zip(padded.taps.chunks_exact(stride))
            {
                let mut sums = [_mm256_setzero_si256(); C];
                let groups = row[x0 * C..(x0 + stride) * C].as_chunks::<8>().0;
                for (eight, groups) in taps
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(groups.as_chunks::<C>().0)
                {
                    let eight = lanes(*eight);
                    for ((sum, group), spread) in sums.iter_mut().zip(groups).zip(&spreads) {
                        let pixels =
                            _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(i64::from_le_bytes(*group)));
                        let taps = _mm256_permutevar8x32_epi32(eight, *spread);
                        *sum = _mm256_add_epi32(*sum, _mm256_mullo_epi32(pixels, taps));
                    }
                }
                *out = clipped::<C>(channel_sums::<C>(&sums), kernel);
            }
            pil_h_outputs::<C>(row, kernel, padded.vector_outputs, rest);
        }
    }

    /// The `C` channel sums of an output pixel in the low `C` lanes, from
    /// the accumulators of its eight-byte groups: lane `l` of accumulator
    /// `q` holds channel `(8q + l) % C`.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn channel_sums<const C: usize>(sums: &[__m256i; C]) -> __m128i {
        const {
            assert!(
                C == 3 || C == 4,
                "the width pass spreads three or four channels"
            );
        }
        match sums.as_slice() {
            [a, b, c] => {
                // Twenty-four lanes whose channel is the lane's index modulo
                // three. Lanes 12.. fold onto lanes 0.. (twelve is a multiple
                // of three): `x` is lanes 0..8 of the fold, channels
                // 0,1,2,0,1,2,0,1; `y` lanes 8..12, channels 2,0,1,2. Then
                // channel 0 is x0 + x3 + x6 + y1, channel 1 x1 + x4 + x7 + y2
                // and channel 2 x2 + x5 + y3 + y0.
                let x = _mm256_add_epi32(*a, _mm256_permute2x128_si256::<0x21>(*b, *c));
                let y = _mm256_zextsi128_si256(_mm_add_epi32(
                    _mm256_castsi256_si128(*b),
                    _mm256_extracti128_si256::<1>(*c),
                ));
                let second = _mm256_permutevar8x32_epi32(x, lanes([3, 4, 5, 0, 0, 0, 0, 0]));
                let third = _mm256_blend_epi32::<0b0000_0100>(
                    _mm256_permutevar8x32_epi32(x, lanes([6, 7, 0, 0, 0, 0, 0, 0])),
                    _mm256_permutevar8x32_epi32(y, lanes([0, 0, 3, 0, 0, 0, 0, 0])),
                );
                let fourth = _mm256_permutevar8x32_epi32(y, lanes([1, 2, 0, 0, 0, 0, 0, 0]));
                _mm256_castsi256_si128(_mm256_add_epi32(
                    _mm256_add_epi32(x, second),
                    _mm256_add_epi32(third, fourth),
                ))
            }
            [a, b, c, d] => {
                // Thirty-two lanes whose channel is the lane's index modulo
                // four: the four accumulators share one lane order, and the
                // high half of their sum folds onto the low half.
                let s = _mm256_add_epi32(_mm256_add_epi32(*a, *b), _mm256_add_epi32(*c, *d));
                _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s))
            }
            // Excluded by the assertion above.
            _ => _mm_setzero_si128(),
        }
    }

    /// `kernel.clip8(half + sum)` of the low `C` lanes, as bytes.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn clipped<const C: usize>(sums: __m128i, kernel: &FixedKernel) -> [u8; C] {
        let shifted = _mm_sra_epi32(
            _mm_add_epi32(sums, _mm_set1_epi32(kernel.half())),
            _mm_cvtsi32_si128(kernel.precision as i32),
        );
        // Both packs saturate: `i32 -> i16` keeps the shifted sums (below 2^10
        // in magnitude), `i16 -> u8` is the clamp to a byte.
        let bytes = _mm_packus_epi16(_mm_packs_epi32(shifted, shifted), _mm_setzero_si128());
        let bytes = _mm_cvtsi128_si32(bytes).to_le_bytes();
        std::array::from_fn(|channel| bytes[channel])
    }

    /// [`pil_v_band`] compiled for AVX2: its row-long inner loops run eight
    /// lanes wide.
    #[target_feature(enable = "avx2")]
    pub(super) fn v_band(
        src: &[u8],
        kernel: &FixedKernel,
        row_len: usize,
        y0: usize,
        band: &mut [u8],
    ) {
        pil_v_band(src, kernel, row_len, y0, band);
    }
}

/// Every row of `src` (`rows` rows of `in_w` interleaved pixels of `C`
/// channels) resampled along the width to `out_w` pixels; rows banded across
/// the pool (an output row depends on its own input row alone, so the banding
/// changes nothing).
fn pil_resample_horizontal<const C: usize>(
    src: &[u8],
    rows: usize,
    in_w: usize,
    out_w: usize,
    kernel: &FixedKernel,
) -> Vec<u8> {
    let row_out = out_w * C;
    let mut out = vec![0_u8; rows * row_out];
    let pass = &WidthPass::new(kernel, in_w);
    let tasks = par_threads(out.len(), rows);
    if tasks <= 1 {
        pass.rows::<C>(src, kernel, in_w, out_w, 0, &mut out);
        return out;
    }
    let band_rows = rows.div_ceil(tasks);
    parallel_scope(|scope| {
        let mut rest = out.as_mut_slice();
        let mut y0 = 0;
        while y0 < rows {
            let n = band_rows.min(rows - y0);
            let (band, tail) = rest.split_at_mut(n * row_out);
            rest = tail;
            let start = y0;
            scope.spawn(move |_| pass.rows::<C>(src, kernel, in_w, out_w, start, band));
            y0 += n;
        }
    });
    out
}

/// `src` (rows of `width` interleaved pixels of `C` channels) resampled along
/// the height to `out_h` rows; rows banded across the pool.
fn pil_resample_vertical<const C: usize>(
    src: &[u8],
    out_h: usize,
    width: usize,
    kernel: &FixedKernel,
) -> Vec<u8> {
    let row_len = width * C;
    let mut out = vec![0_u8; out_h * row_len];
    let tasks = par_threads(out.len(), out_h);
    if tasks <= 1 {
        pil_v_rows(src, kernel, row_len, 0, &mut out);
        return out;
    }
    let band_rows = out_h.div_ceil(tasks);
    parallel_scope(|scope| {
        let mut rest = out.as_mut_slice();
        let mut y0 = 0;
        while y0 < out_h {
            let n = band_rows.min(out_h - y0);
            let (band, tail) = rest.split_at_mut(n * row_len);
            rest = tail;
            let start = y0;
            scope.spawn(move |_| pil_v_rows(src, kernel, row_len, start, band));
            y0 += n;
        }
    });
    out
}

/// Pillow-exact BICUBIC resize (RGB8), matching
/// `PIL.Image.resize(.., BICUBIC)`.
pub fn resize_bicubic_pil(image: &DynamicImage, out_w: u32, out_h: u32) -> DynamicImage {
    let rgb = image.to_rgb8();
    let (in_w, in_h) = rgb.dimensions();
    let output = resize_pil_bytes::<3>(
        rgb.as_raw(),
        in_w,
        in_h,
        out_w,
        out_h,
        PilResizeFilter::Bicubic,
        KernelQuantization::Pillow,
    );
    #[expect(
        clippy::expect_used,
        reason = "output is exactly out_w*out_h*3 bytes by construction"
    )]
    DynamicImage::ImageRgb8(
        RgbImage::from_raw(out_w, out_h, output).expect("pil resize buffer size"),
    )
}

/// PIL-exact bicubic resize over borrowed interleaved RGB bytes.
///
/// Byte-for-byte equivalent of [`resize_bicubic_pil`] but for the raw-RGB video
/// frame path (`preprocess_video_rgb`). Returns an `RgbImage` to drop straight
/// into the existing [`resize_rgb_bytes`] call sites.
pub fn resize_bicubic_pil_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    out_w: u32,
    out_h: u32,
) -> Result<RgbImage> {
    let (in_w, in_h) = (width as usize, height as usize);
    let expected = in_w.saturating_mul(in_h).saturating_mul(3);
    if data.len() != expected {
        return Err(TransformError::ShapeError(format!(
            "PIL bicubic RGB source has {} bytes, expected {expected} for {width}x{height}",
            data.len()
        )));
    }
    if out_w == 0 || out_h == 0 {
        return Err(TransformError::ShapeError(format!(
            "PIL bicubic RGB target {out_w}x{out_h} must be non-empty"
        )));
    }
    let output = resize_pil_bytes::<3>(
        data,
        width,
        height,
        out_w,
        out_h,
        PilResizeFilter::Bicubic,
        KernelQuantization::Pillow,
    );
    RgbImage::from_raw(out_w, out_h, output).ok_or_else(|| {
        TransformError::ShapeError(format!(
            "failed to build PIL bicubic RGB image for {out_w}x{out_h}"
        ))
    })
}

/// Pillow-exact LANCZOS resize (RGB8), matching
/// `PIL.Image.resize(.., LANCZOS)`.
pub fn resize_lanczos_pil(image: &DynamicImage, out_w: u32, out_h: u32) -> DynamicImage {
    let rgb = image.to_rgb8();
    let (in_w, in_h) = rgb.dimensions();
    let output = resize_pil_bytes::<3>(
        rgb.as_raw(),
        in_w,
        in_h,
        out_w,
        out_h,
        PilResizeFilter::Lanczos,
        KernelQuantization::Pillow,
    );
    #[expect(
        clippy::expect_used,
        reason = "output is exactly out_w*out_h*3 bytes by construction"
    )]
    DynamicImage::ImageRgb8(
        RgbImage::from_raw(out_w, out_h, output).expect("pil resize buffer size"),
    )
}

/// `data` (`in_w` x `in_h` interleaved pixels of `C` channels) resampled to
/// `out_w` x `out_h` with `filter`'s kernel in `quantization`: the width pass
/// first, then the height pass, each skipped on an axis that keeps its size
/// (Pillow's own order and skips). `C` is 3 or 4, the channel counts the
/// vector arms' lane folds spread; the rule holds on every platform.
fn resize_pil_bytes<const C: usize>(
    data: &[u8],
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    filter: PilResizeFilter,
    quantization: KernelQuantization,
) -> Vec<u8> {
    const {
        assert!(
            C == 3 || C == 4,
            "the Pillow-exact resample takes three or four channels"
        );
    }
    let (in_w, in_h, out_w, out_h) = (in_w as usize, in_h as usize, out_w as usize, out_h as usize);
    if out_w == 0 || out_h == 0 {
        // A zero-sized target has no pixels; the band resamplers would panic
        // on a zero chunk size. The fallible entry points
        // (`resize_bicubic_pil_rgb`, `pad_to_size_pil_rgb`) reject it up
        // front; the infallible ones are only called with planned, non-zero
        // sizes.
        Vec::new()
    } else if in_w == out_w && in_h == out_h {
        data.to_vec()
    } else if in_w == out_w {
        let kernel = FixedKernel::new(in_h, out_h, filter, quantization);
        pil_resample_vertical::<C>(data, out_h, in_w, &kernel)
    } else {
        let kernel = FixedKernel::new(in_w, out_w, filter, quantization);
        let horizontal = pil_resample_horizontal::<C>(data, in_h, in_w, out_w, &kernel);
        if in_h == out_h {
            horizontal
        } else {
            let kernel = FixedKernel::new(in_h, out_h, filter, quantization);
            pil_resample_vertical::<C>(&horizontal, out_h, out_w, &kernel)
        }
    }
}

/// PyTorch's antialiased BICUBIC resize of `uint8` RGB pixels: what
/// `torch.nn.functional.interpolate(mode="bicubic", antialias=True)` computes
/// on a CPU `uint8` tensor, which is where torchvision's `resize` sends a
/// `uint8` image and so what transformers' torchvision-backed image
/// processors (the ones vLLM runs) do to the decoded bytes.
///
/// Pillow's separable resample (width first, then height, each pass rounded
/// back to bytes) with one difference from Pillow's own current code: the
/// normalized kernel is quantized to 16-bit taps at the finest scale that
/// keeps the largest weight below 2^15, not at a fixed 22 bits
/// (`KernelQuantization::TorchUint8`). The two differ by up to two levels on
/// a fraction of the pixels, which an engine fed by such a processor sees.
/// The passes run in `super::torch_resample`: 16-bit taps, 32-bit sums, on
/// the vector unit where there is one.
pub fn resize_bicubic_torch_aa_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    out_w: u32,
    out_h: u32,
) -> Result<RgbImage> {
    let (in_w, in_h) = (width as usize, height as usize);
    let expected = in_w.saturating_mul(in_h).saturating_mul(3);
    if data.len() != expected {
        return Err(TransformError::ShapeError(format!(
            "torch bicubic RGB source has {} bytes, expected {expected} for {width}x{height}",
            data.len()
        )));
    }
    if width == 0 || height == 0 || out_w == 0 || out_h == 0 {
        return Err(TransformError::ShapeError(format!(
            "torch bicubic RGB resize {width}x{height} -> {out_w}x{out_h} must be non-empty"
        )));
    }
    let output = super::torch_resample::resize_rgb(
        data,
        in_w,
        in_h,
        out_w as usize,
        out_h as usize,
        PilResizeFilter::Bicubic,
    );
    RgbImage::from_raw(out_w, out_h, output).ok_or_else(|| {
        TransformError::ShapeError(format!(
            "failed to build torch bicubic RGB image for {out_w}x{out_h}"
        ))
    })
}

/// The torchvision `uint8` LANCZOS resize (`interpolate(.., mode="lanczos",
/// antialias=True)` on a `uint8` tensor, what torchvision 0.27 and later
/// run for `InterpolationMode.LANCZOS` on the CPU): Pillow's Lanczos-3
/// kernel in the torch `uint8` quantization of [`resize_bicubic_torch_aa_rgb`],
/// interleaved RGB bytes in and out.
pub fn resize_lanczos_torch_aa_rgb_bytes(
    data: &[u8],
    width: u32,
    height: u32,
    out_w: u32,
    out_h: u32,
) -> Result<Vec<u8>> {
    let (in_w, in_h) = (width as usize, height as usize);
    let expected = in_w.saturating_mul(in_h).saturating_mul(3);
    if data.len() != expected {
        return Err(TransformError::ShapeError(format!(
            "torch lanczos RGB source has {} bytes, expected {expected} for {width}x{height}",
            data.len()
        )));
    }
    if width == 0 || height == 0 || out_w == 0 || out_h == 0 {
        return Err(TransformError::ShapeError(format!(
            "torch lanczos RGB resize {width}x{height} -> {out_w}x{out_h} must be non-empty"
        )));
    }
    Ok(super::torch_resample::resize_rgb(
        data,
        in_w,
        in_h,
        out_w as usize,
        out_h as usize,
        PilResizeFilter::Lanczos,
    ))
}

/// The torchvision `uint8` BICUBIC resize of [`resize_bicubic_torch_aa_rgb`]
/// for a decoded image.
pub fn resize_bicubic_torch_aa(image: &DynamicImage, out_w: u32, out_h: u32) -> DynamicImage {
    let (in_w, in_h, data) = rgb_bytes(image);
    let output = super::torch_resample::resize_rgb(
        &data,
        in_w,
        in_h,
        out_w as usize,
        out_h as usize,
        PilResizeFilter::Bicubic,
    );
    #[expect(
        clippy::expect_used,
        reason = "output is exactly out_w*out_h*3 bytes by construction"
    )]
    DynamicImage::ImageRgb8(
        RgbImage::from_raw(out_w, out_h, output).expect("torch resize buffer size"),
    )
}

/// Keys' cubic (`a = -0.5`) as torch's compiled CPU kernel evaluates it in
/// `f32`: `((a + 2) x - (a + 3)) x x + 1` below one and
/// `((a x - 5a) x + 8a) x - 4a` below two, with the fused multiply-adds its
/// compiler contracts (the inner `a x - c` and every trailing `t x + c`).
/// The same polynomials in double, or in `f32` without the contractions,
/// move a weight by one `f32` ULP at non-dyadic ratios and the output with it.
fn torch_aa_cubic_f32(x: f32) -> f32 {
    const A: f32 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0).mul_add(x, -(A + 3.0)) * x).mul_add(x, 1.0)
    } else if x < 2.0 {
        A.mul_add(x, -(5.0 * A))
            .mul_add(x, 8.0 * A)
            .mul_add(x, -(4.0 * A))
    } else {
        0.0
    }
}

/// torch's antialiasing bicubic weights for one axis, computed as its CPU
/// kernel does on a float tensor: the scale, kernel center and normalized
/// weights in `f32`, the cubic of [`torch_aa_cubic_f32`], and the tap count
/// read from the rounded bounds and capped at the kernel's weight buffer
/// (`ceil(support) * 2 + 1` taps: the rounded bounds can count one more).
/// Pillow's `f64` weights shift by up to ~5e-5 at large output indices, which
/// moves sharp edges by ~0.01.
fn torch_aa_bicubic_weights(
    in_size: usize,
    out_size: usize,
) -> (Vec<(usize, usize)>, Vec<Vec<f32>>) {
    let scale = in_size as f32 / out_size as f32;
    let (support, invscale) = if scale >= 1.0 {
        (2.0 * scale, 1.0 / scale)
    } else {
        (2.0, 1.0)
    };
    let max_taps = (support.ceil() as usize) * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = Vec::with_capacity(out_size);
    for i in 0..out_size {
        let center = (f64::from(scale) * (i as f64 + 0.5)) as f32;
        let start = (f64::from(center - support) + 0.5).max(0.0) as usize;
        let end = ((f64::from(center + support) + 0.5) as usize).min(in_size);
        let count = end.saturating_sub(start).min(max_taps);
        let mut w: Vec<f32> = (start..start + count)
            .map(|j| {
                let x = (f64::from(j as f32 - center) + 0.5) * f64::from(invscale);
                torch_aa_cubic_f32(x as f32)
            })
            .collect();
        let total: f32 = w.iter().sum();
        if total != 0.0 {
            for wx in &mut w {
                *wx /= total;
            }
        }
        bounds.push((start, count));
        weights.push(w);
    }
    (bounds, weights)
}

/// The taps after the first that torch's compiled loop adds plainly for an
/// output of `count` taps: its unrolled body takes four taps at a time after
/// the first, a rounded product and a rounded add each; the remaining taps
/// (the loop's tail, contracted by its compiler) are fused multiply-adds.
/// Read from the kernel's output on impulse inputs, tap count by tap count;
/// every other order differs in the last bit on some inputs.
#[inline]
fn torch_aa_plain_until(count: usize) -> usize {
    (count.saturating_sub(1) / 4) * 4
}

/// `torch.nn.functional.interpolate(mode="bicubic", antialias=True)` on the
/// image as a float tensor: interleaved RGB bytes in, interleaved `f32`
/// values on the byte scale out, nothing rounded. The kernel is Keys cubic
/// `a = -0.5` widened by the downscale factor, with torch's `f32` weights and
/// its compiled loop's order of roundings (`torch_aa_passes`), so an output is
/// the reference's bit for bit; the width pass runs first.
pub fn resize_bicubic_torch_aa_f32(
    data: &[u8],
    width: u32,
    height: u32,
    out_w: u32,
    out_h: u32,
) -> Result<Vec<f32>> {
    let (in_w, in_h, out_w, out_h) = (
        width as usize,
        height as usize,
        out_w as usize,
        out_h as usize,
    );
    if data.len() != in_w.saturating_mul(in_h).saturating_mul(3) {
        return Err(TransformError::ShapeError(format!(
            "torch bicubic RGB source has {} bytes, expected {} for {width}x{height}",
            data.len(),
            in_w * in_h * 3
        )));
    }
    if in_w == 0 || in_h == 0 || out_w == 0 || out_h == 0 {
        return Err(TransformError::ShapeError(format!(
            "torch bicubic RGB resize {width}x{height} -> {out_w}x{out_h} must be non-empty"
        )));
    }
    let (x_bounds, x_weights) = torch_aa_bicubic_weights(in_w, out_w);
    let (y_bounds, y_weights) = torch_aa_bicubic_weights(in_h, out_h);
    let x = TorchAaAxis {
        bounds: &x_bounds,
        weights: &x_weights,
    };
    let y = TorchAaAxis {
        bounds: &y_bounds,
        weights: &y_weights,
    };
    #[cfg(target_arch = "x86_64")]
    if let Some(out) = torch_aa_passes_with_fma(data, in_w, &x, &y) {
        return Ok(out);
    }
    Ok(torch_aa_passes(data, in_w, &x, &y))
}

/// One axis of the torch-style resize: per output sample the first source
/// index and the tap count, and the taps' weights.
struct TorchAaAxis<'a> {
    bounds: &'a [(usize, usize)],
    weights: &'a [Vec<f32>],
}

/// The two passes of [`resize_bicubic_torch_aa_f32`]: every source row
/// resampled along the width, then the rows along the height. An output's
/// taps add in the reference's three runs: the first tap is the bare product,
/// the next [`torch_aa_plain_until`] taps add a rounded product in a second
/// rounding, and the rest are fused multiply-adds. Inlined into its callers so
/// that the fused taps compile to the caller's instruction set.
#[inline(always)]
fn torch_aa_passes(data: &[u8], in_w: usize, x: &TorchAaAxis<'_>, y: &TorchAaAxis<'_>) -> Vec<f32> {
    let (out_w, out_h) = (x.bounds.len(), y.bounds.len());
    let in_h = data.len() / (in_w * 3);

    let mut rows = vec![0.0_f32; in_h * out_w * 3];
    for (src, dst) in data
        .chunks_exact(in_w * 3)
        .zip(rows.chunks_exact_mut(out_w * 3))
    {
        let (pixels, _) = src.as_chunks::<3>();
        let (outs, _) = dst.as_chunks_mut::<3>();
        for (out, (&(start, count), weights)) in outs.iter_mut().zip(x.bounds.iter().zip(x.weights))
        {
            let mut taps = weights[..count].iter().zip(&pixels[start..start + count]);
            let mut acc = [0.0_f32; 3];
            if let Some((&weight, pixel)) = taps.next() {
                for (sum, &byte) in acc.iter_mut().zip(pixel) {
                    *sum = weight * f32::from(byte);
                }
            }
            for (&weight, pixel) in taps.by_ref().take(torch_aa_plain_until(count)) {
                for (sum, &byte) in acc.iter_mut().zip(pixel) {
                    *sum += weight * f32::from(byte);
                }
            }
            for (&weight, pixel) in taps {
                for (sum, &byte) in acc.iter_mut().zip(pixel) {
                    *sum = weight.mul_add(f32::from(byte), *sum);
                }
            }
            *out = acc;
        }
    }

    let mut out = vec![0.0_f32; out_h * out_w * 3];
    let row = out_w * 3;
    for (dst, (&(start, count), weights)) in out
        .chunks_exact_mut(row)
        .zip(y.bounds.iter().zip(y.weights))
    {
        let mut taps = weights[..count]
            .iter()
            .zip(rows[start * row..(start + count) * row].chunks_exact(row));
        if let Some((&weight, src)) = taps.next() {
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = weight * v;
            }
        }
        for (&weight, src) in taps.by_ref().take(torch_aa_plain_until(count)) {
            for (d, &v) in dst.iter_mut().zip(src) {
                *d += weight * v;
            }
        }
        for (&weight, src) in taps {
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = weight.mul_add(v, *d);
            }
        }
    }
    out
}

/// [`torch_aa_passes`] compiled for the fused multiply-add instruction. At
/// the baseline x86-64 target the fused taps of [`torch_aa_passes`] call the C
/// library's software `fmaf`, the same bits several times slower; the
/// reference kernel dispatches a build of its own with the instruction on
/// such processors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
fn torch_aa_passes_fma(
    data: &[u8],
    in_w: usize,
    x: &TorchAaAxis<'_>,
    y: &TorchAaAxis<'_>,
) -> Vec<f32> {
    torch_aa_passes(data, in_w, x, y)
}

/// [`torch_aa_passes_fma`] on a processor that reports the feature, `None`
/// on one that does not: the detection and the call sit in one function,
/// as the crate's NEON kernels keep them.
#[cfg(target_arch = "x86_64")]
#[expect(
    unsafe_code,
    reason = "the target-feature function runs only after this function detected its feature"
)]
fn torch_aa_passes_with_fma(
    data: &[u8],
    in_w: usize,
    x: &TorchAaAxis<'_>,
    y: &TorchAaAxis<'_>,
) -> Option<Vec<f32>> {
    if !std::arch::is_x86_feature_detected!("fma") {
        return None;
    }
    // SAFETY: `is_x86_feature_detected!("fma")` answered true just above, the
    // only requirement of a `target_feature(enable = "fma")` function.
    Some(unsafe { torch_aa_passes_fma(data, in_w, x, y) })
}

/// Resize image preserving aspect ratio, fitting within max dimensions.
pub fn resize_to_fit(
    image: &DynamicImage,
    max_width: u32,
    max_height: u32,
    filter: FilterType,
) -> DynamicImage {
    let (w, h) = image.dimensions();
    let ratio = (max_width as f64 / w as f64).min(max_height as f64 / h as f64);
    if ratio >= 1.0 {
        return image.clone();
    }
    let new_w = ((w as f64 * ratio).round() as u32).max(1);
    let new_h = ((h as f64 * ratio).round() as u32).max(1);
    resize(image, new_w, new_h, filter)
}

/// Center crop image to specified dimensions.
///
/// If the crop size is larger than the image, the image is returned unchanged.
pub fn center_crop(image: &DynamicImage, crop_w: u32, crop_h: u32) -> DynamicImage {
    let (w, h) = image.dimensions();
    if crop_w >= w && crop_h >= h {
        return image.clone();
    }
    let left = (w.saturating_sub(crop_w)) / 2;
    let top = (h.saturating_sub(crop_h)) / 2;
    let actual_w = crop_w.min(w);
    let actual_h = crop_h.min(h);
    match image {
        DynamicImage::ImageRgb8(rgb) => {
            DynamicImage::ImageRgb8(crop_rgb(rgb, left, top, actual_w, actual_h))
        }
        other => other.crop_imm(left, top, actual_w, actual_h),
    }
}

/// The `width` x `height` crop of `image` at (`x`, `y`), clipped to the
/// image as the generic crop clips it, as row copies out of the image's own
/// bytes. The generic crop builds a zeroed image and copies pixel by pixel;
/// the zero fill cost the allocator's page faults on a crop of the image's
/// own scale, and the per-pixel copy its bounds checks.
pub(crate) fn crop_rgb(image: &RgbImage, x: u32, y: u32, width: u32, height: u32) -> RgbImage {
    let (image_w, image_h) = image.dimensions();
    let (x, y) = (x.min(image_w), y.min(image_h));
    let (width, height) = (width.min(image_w - x), height.min(image_h - y));
    // A zero-width image has no rows; a chunk size of 0 would panic.
    let stride = (image_w as usize * 3).max(1);
    let (start, len) = (x as usize * 3, width as usize * 3);
    let mut data = Vec::with_capacity(len * height as usize);
    for row in image
        .as_raw()
        .chunks_exact(stride)
        .skip(y as usize)
        .take(height as usize)
    {
        data.extend_from_slice(&row[start..start + len]);
    }
    #[expect(
        clippy::expect_used,
        reason = "data holds exactly width * height * 3 bytes by construction"
    )]
    RgbImage::from_raw(width, height, data).expect("crop buffer size")
}

/// `image` pasted at (`x`, `y`) on a `width` x `height` canvas of
/// `background`, built row by row: background rows above and below the
/// paste, and every pasted row the left margin, the image's row and the
/// right margin. The part of the image past the canvas's edges is left out,
/// as the generic paste leaves it out; the paste of an opaque image is a
/// copy. The canvas is written exactly once; a canvas filled first and then
/// painted pixel by pixel paid the fill (and, for a zeroed canvas, the
/// allocator's page faults) and the paste's bounds checks per pixel.
pub(crate) fn paste_rgb(
    image: &RgbImage,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    background: [u8; 3],
) -> RgbImage {
    let (visible_w, visible_h) = (
        image.width().min(width.saturating_sub(x)) as usize,
        image.height().min(height.saturating_sub(y)) as usize,
    );
    let row = width as usize * 3;
    let left = x.min(width) as usize * 3;
    let right = row - left - visible_w * 3;
    let background_row: Vec<u8> = background.repeat(width as usize);
    let mut canvas = Vec::with_capacity(row * height as usize);
    for _ in 0..y.min(height) {
        canvas.extend_from_slice(&background_row);
    }
    // A zero-width image has no rows to paste; a chunk size of 0 would panic.
    for source_row in image
        .as_raw()
        .chunks_exact((image.width() as usize * 3).max(1))
        .take(visible_h)
    {
        canvas.extend_from_slice(&background_row[..left]);
        canvas.extend_from_slice(&source_row[..visible_w * 3]);
        canvas.extend_from_slice(&background_row[..right]);
    }
    while canvas.len() < row * height as usize {
        canvas.extend_from_slice(&background_row);
    }
    #[expect(
        clippy::expect_used,
        reason = "the canvas holds exactly width * height * 3 bytes by construction"
    )]
    RgbImage::from_raw(width, height, canvas).expect("canvas buffer size")
}

/// Expand image to square by padding with background color.
///
/// This is used by LLaVA models which expect square inputs. The image is
/// centered and padded with the mean color on the shorter dimension.
pub fn expand_to_square(image: &DynamicImage, background: Rgb<u8>) -> DynamicImage {
    let (w, h) = image.dimensions();
    match w.cmp(&h) {
        std::cmp::Ordering::Equal => image.clone(),
        std::cmp::Ordering::Less => {
            // Height > Width: pad horizontally
            let mut new_image = DynamicImage::from(RgbImage::from_pixel(h, h, background));
            image::imageops::overlay(&mut new_image, image, ((h - w) / 2) as i64, 0);
            new_image
        }
        std::cmp::Ordering::Greater => {
            // Width > Height: pad vertically
            let mut new_image = DynamicImage::from(RgbImage::from_pixel(w, w, background));
            image::imageops::overlay(&mut new_image, image, 0, ((w - h) / 2) as i64);
            new_image
        }
    }
}

/// Pillow-exact `ImageOps.pad` with default centering (0.5, 0.5):
/// aspect-preserving BICUBIC `contain` fit into `(out_w, out_h)`, then a
/// centered paste onto a `color` canvas. Matches Pillow 12.x source: `contain`
/// adjusts at most one dimension using Python `round()` (exact
/// round-half-to-even, so `471.49999999999994` rounds to 471 like Python and
/// not up like an epsilon tie test would) and `pad` pastes at
/// `round((size - resized) * 0.5)` along the padded axis.
///
/// The image is treated as RGB (`to_rgb8`): an alpha channel is dropped, not
/// premultiplied; callers that need Pillow's RGBA behaviour must flatten
/// first. An aspect ratio so extreme that the contained dimension rounds to
/// zero is an error, as it is in Pillow (`height and width must be > 0`).
pub fn pad_to_size_pil(
    image: &DynamicImage,
    out_w: u32,
    out_h: u32,
    color: Rgb<u8>,
) -> Result<DynamicImage> {
    let (w, h, data) = rgb_bytes(image);
    pad_to_size_pil_rgb(&data, w as u32, h as u32, out_w, out_h, color).map(DynamicImage::ImageRgb8)
}

/// [`pad_to_size_pil`] over interleaved RGB bytes (`width` x `height`): the
/// same geometry, the contained image resized once and its rows copied into
/// the canvas (the paste of an opaque image is a copy).
pub fn pad_to_size_pil_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    out_w: u32,
    out_h: u32,
    color: Rgb<u8>,
) -> Result<RgbImage> {
    let expected = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(3);
    if data.len() != expected {
        return Err(TransformError::ShapeError(format!(
            "pad source has {} bytes, expected {expected} for {width}x{height}",
            data.len()
        )));
    }
    let (w, h) = (width, height);
    let (mut target_w, mut target_h) = (out_w, out_h);

    // ImageOps.contain: fit within (out_w, out_h) preserving aspect ratio.
    let im_ratio = f64::from(w) / f64::from(h);
    let dest_ratio = f64::from(out_w) / f64::from(out_h);
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            let new_h = (f64::from(h) / f64::from(w) * f64::from(out_w)).round_ties_even() as u32;
            if new_h != out_h {
                target_h = new_h;
            }
        } else {
            let new_w = (f64::from(w) / f64::from(h) * f64::from(out_h)).round_ties_even() as u32;
            if new_w != out_w {
                target_w = new_w;
            }
        }
    }
    if target_w == 0 || target_h == 0 {
        return Err(TransformError::ShapeError(format!(
            "contain fit of {w}x{h} into {out_w}x{out_h} has a zero dimension ({target_w}x{target_h})"
        )));
    }

    let resized = resize_pil_bytes::<3>(
        data,
        w,
        h,
        target_w,
        target_h,
        PilResizeFilter::Bicubic,
        KernelQuantization::Pillow,
    );
    let canvas = if target_w == out_w && target_h == out_h {
        resized
    } else {
        let (canvas_w, canvas_h) = (out_w as usize, out_h as usize);
        let (resized_w, resized_h) = (target_w as usize, target_h as usize);
        let mut canvas = vec![0_u8; canvas_w * canvas_h * 3];
        for pixel in canvas.as_chunks_mut::<3>().0.iter_mut() {
            *pixel = color.0;
        }
        if resized_w == canvas_w {
            let y = (f64::from(out_h - target_h) * 0.5).round_ties_even() as usize;
            canvas[y * canvas_w * 3..(y + resized_h) * canvas_w * 3].copy_from_slice(&resized);
        } else {
            let x = (f64::from(out_w - target_w) * 0.5).round_ties_even() as usize;
            for (row, resized_row) in canvas
                .chunks_exact_mut(canvas_w * 3)
                .zip(resized.chunks_exact(resized_w * 3))
            {
                row[x * 3..(x + resized_w) * 3].copy_from_slice(resized_row);
            }
        }
        canvas
    };
    RgbImage::from_raw(out_w, out_h, canvas).ok_or_else(|| {
        TransformError::ShapeError(format!("failed to build the padded {out_w}x{out_h} image"))
    })
}

/// Stack multiple [C, H, W] tensors into [B, C, H, W].
///
/// All tensors must have the same shape.
pub fn stack_batch(tensors: &[Array3<f32>]) -> Result<Array4<f32>> {
    if tensors.is_empty() {
        return Err(TransformError::EmptyBatch);
    }

    let shape = tensors[0].shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);

    // Verify all tensors have the same shape
    for tensor in tensors.iter().skip(1) {
        if tensor.shape() != shape {
            return Err(TransformError::InvalidShape {
                expected: format!("[{c}, {h}, {w}]"),
                actual: tensor.shape().to_vec(),
            });
        }
    }

    // The batch assembled from the tensors by copy, every element written
    // once; a zeroed batch assigned tensor by tensor paid the allocator's page
    // faults for a zero fill the copies then overwrote.
    let mut data = Vec::with_capacity(tensors.len() * c * h * w);
    for tensor in tensors {
        match tensor.as_slice() {
            Some(values) => data.extend_from_slice(values),
            None => data.extend(tensor.iter()),
        }
    }
    #[expect(
        clippy::expect_used,
        reason = "data holds tensors.len() * c * h * w elements by construction"
    )]
    Ok(Array4::from_shape_vec((tensors.len(), c, h, w), data).expect("batch matches its shape"))
}

/// Convert PIL/HuggingFace resampling enum to image crate filter.
///
/// PIL resampling constants:
/// - 0: NEAREST
/// - 1: LANCZOS (also ANTIALIAS)
/// - 2: BILINEAR
/// - 3: BICUBIC
/// - 4: BOX
/// - 5: HAMMING
pub fn pil_to_filter(resampling: Option<usize>) -> FilterType {
    match resampling {
        Some(0) => FilterType::Nearest,
        Some(1) => FilterType::Lanczos3,
        Some(2) | None => FilterType::Triangle, // Bilinear (default)
        Some(3) => FilterType::CatmullRom,      // Bicubic
        // Box and Hamming don't have direct equivalents, use Triangle
        Some(4) | Some(5) => FilterType::Triangle,
        _ => FilterType::Triangle,
    }
}

/// Calculate mean color of an image as RGB.
pub fn calculate_mean_color(image: &DynamicImage) -> Rgb<u8> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as u64, rgb.height() as u64);
    let total_pixels = w * h;

    if total_pixels == 0 {
        return Rgb([128, 128, 128]);
    }

    let (mut r_sum, mut g_sum, mut b_sum) = (0u64, 0u64, 0u64);
    for pixel in rgb.pixels() {
        r_sum += pixel[0] as u64;
        g_sum += pixel[1] as u64;
        b_sum += pixel[2] as u64;
    }

    Rgb([
        (r_sum / total_pixels) as u8,
        (g_sum / total_pixels) as u8,
        (b_sum / total_pixels) as u8,
    ])
}

/// Convert normalized mean values [0, 1] to RGB bytes.
pub fn mean_to_rgb(mean: &[f64; 3]) -> Rgb<u8> {
    Rgb([
        (mean[0] * 255.0).round() as u8,
        (mean[1] * 255.0).round() as u8,
        (mean[2] * 255.0).round() as u8,
    ])
}

/// Cubic interpolation weight function (Keys bicubic kernel with a=-0.5).
///
/// This matches PyTorch's bicubic interpolation used in
/// `torch.nn.functional.interpolate(mode='bicubic')`.
#[inline]
pub fn cubic_weight(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Perform bicubic interpolation at a single point in a tensor.
///
/// Uses a 4x4 kernel with Keys bicubic weights (a=-0.5) to match PyTorch's
/// `torch.nn.functional.interpolate(mode='bicubic')`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [C, H, W]
/// * `c` - Channel index
/// * `src_y` - Source Y coordinate (can be fractional)
/// * `src_x` - Source X coordinate (can be fractional)
/// * `h` - Height of the tensor
/// * `w` - Width of the tensor
///
/// # Returns
/// The interpolated value at the specified position.
pub fn bicubic_interpolate(
    tensor: &Array3<f32>,
    c: usize,
    src_y: f32,
    src_x: f32,
    h: usize,
    w: usize,
) -> f32 {
    let y_int = src_y.floor() as i32;
    let x_int = src_x.floor() as i32;
    let y_frac = src_y - y_int as f32;
    let x_frac = src_x - x_int as f32;

    let mut result = 0.0f32;

    // Sample 4x4 neighborhood
    for dy in -1..=2 {
        let y_idx = (y_int + dy).clamp(0, h as i32 - 1) as usize;
        let y_weight = cubic_weight(y_frac - dy as f32);

        for dx in -1..=2 {
            let x_idx = (x_int + dx).clamp(0, w as i32 - 1) as usize;
            let x_weight = cubic_weight(x_frac - dx as f32);

            result += tensor[[c, y_idx, x_idx]] * y_weight * x_weight;
        }
    }

    result
}

/// Resize a tensor using bicubic interpolation.
///
/// This matches PyTorch's `torch.nn.functional.interpolate(mode='bicubic', align_corners=False)`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [C, H, W]
/// * `target_h` - Target height
/// * `target_w` - Target width
///
/// # Returns
/// Resized tensor of shape [C, target_h, target_w].
pub fn bicubic_resize(tensor: &Array3<f32>, target_h: usize, target_w: usize) -> Array3<f32> {
    let (c, h, w) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);

    if h == target_h && w == target_w {
        return tensor.clone();
    }

    let mut result = Array3::<f32>::zeros((c, target_h, target_w));

    // PyTorch align_corners=False coordinate mapping
    let scale_h = h as f32 / target_h as f32;
    let scale_w = w as f32 / target_w as f32;

    for ch in 0..c {
        for y in 0..target_h {
            for x in 0..target_w {
                // PyTorch align_corners=False: src = (dst + 0.5) * scale - 0.5
                let src_y = (y as f32 + 0.5) * scale_h - 0.5;
                let src_x = (x as f32 + 0.5) * scale_w - 0.5;

                result[[ch, y, x]] = bicubic_interpolate(tensor, ch, src_y, src_x, h, w);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {

    use image::Rgba;

    use super::*;
    use crate::test_support::{assert_same_bytes, create_test_image, gradient_rgb};

    #[test]
    fn catmull_rom_rgb_matches_image_resize() {
        for (width, height, out_w, out_h) in [
            (1, 17, 1, 7),
            (17, 1, 7, 1),
            (19, 13, 7, 5),
            (97, 53, 28, 14),
            (101, 77, 100, 76),
            (64, 64, 64, 64),
            (2240, 280, 2016, 252),
            (3840, 2160, 2016, 1134),
        ] {
            let mut seed = 17u32;
            let input = RgbImage::from_fn(width, height, |_, _| {
                Rgb(std::array::from_fn(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 24) as u8
                }))
            });
            let expected = image::imageops::resize(&input, out_w, out_h, FilterType::CatmullRom);
            let actual = resize_catmull_rom_rgb(&input, out_w, out_h);
            assert_eq!(
                actual.as_raw(),
                expected.as_raw(),
                "{width}x{height} -> {out_w}x{out_h}"
            );
            assert_eq!(actual.color_space(), expected.color_space());
        }
    }

    #[test]
    fn pad_to_size_pil_centres_on_one_axis_only() {
        // 64x48 into 630x476: width-limited, so the contain fit is 630x472
        // (472.5 rounds half to even) and a 2-row gray band goes on top.
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 48, Rgb([10, 200, 30])));
        let out = pad_to_size_pil(&img, 630, 476, Rgb([127, 127, 127]))
            .unwrap()
            .to_rgb8();
        assert_eq!(out.dimensions(), (630, 476));
        assert_eq!(out.get_pixel(0, 0).0, [127, 127, 127]);
        assert_eq!(out.get_pixel(0, 475).0, [127, 127, 127]);
        assert_eq!(out.get_pixel(315, 238).0, [10, 200, 30]);
        // Same aspect: no padding, a plain bicubic resize.
        let same = pad_to_size_pil(&img, 128, 96, Rgb([0, 0, 0]))
            .unwrap()
            .to_rgb8();
        assert_eq!(same.dimensions(), (128, 96));
        assert_eq!(same.get_pixel(0, 0).0, [10, 200, 30]);
    }

    #[test]
    fn pad_to_size_pil_rounds_the_contained_edge_like_python() {
        // 41x56 into 476x644: 41/56*644 = 471.49999999999994, which Python's
        // round() takes to 471 (an epsilon tie test would say 472) and then
        // pastes at x = round(2.5) = 2.
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(41, 56, Rgb([200, 10, 30])));
        let out = pad_to_size_pil(&img, 476, 644, Rgb([127, 127, 127]))
            .unwrap()
            .to_rgb8();
        assert_eq!(out.dimensions(), (476, 644));
        assert_eq!(out.get_pixel(1, 300).0, [127, 127, 127]);
        assert_eq!(out.get_pixel(2, 300).0, [200, 10, 30]);
        assert_eq!(out.get_pixel(472, 300).0, [200, 10, 30]);
        assert_eq!(out.get_pixel(473, 300).0, [127, 127, 127]);
    }

    #[test]
    fn pad_to_size_pil_rejects_a_zero_contained_dimension() {
        // Pillow raises for these: the contained width/height rounds to 0.
        let tall = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 50_000, Rgb([0, 0, 0])));
        assert!(matches!(
            pad_to_size_pil(&tall, 42, 21_462, Rgb([127, 127, 127])),
            Err(TransformError::ShapeError(_))
        ));
        let wide = DynamicImage::ImageRgb8(RgbImage::from_pixel(100_000, 1, Rgb([0, 0, 0])));
        assert!(matches!(
            pad_to_size_pil(&wide, 42_882, 42, Rgb([127, 127, 127])),
            Err(TransformError::ShapeError(_))
        ));
    }

    /// The raw-RGB torchvision `uint8` resizer must be byte-for-byte identical
    /// to the `DynamicImage` form the Qwen image path uses (the raw-RGB form
    /// serves `preprocess_video_rgb`).
    #[test]
    fn resize_bicubic_torch_aa_rgb_matches_dynamic_path() {
        let (src_w, src_h) = (37u32, 23u32);
        let (out_w, out_h) = (16u32, 28u32);
        let mut img = RgbImage::new(src_w, src_h);
        for y in 0..src_h {
            for x in 0..src_w {
                img.put_pixel(
                    x,
                    y,
                    Rgb([
                        ((x * 7) ^ (y * 13)) as u8,
                        (x * 3 + y * 5) as u8,
                        (x + y * y) as u8,
                    ]),
                );
            }
        }
        let via_dynamic =
            resize_bicubic_torch_aa(&DynamicImage::ImageRgb8(img.clone()), out_w, out_h);
        let via_bytes =
            resize_bicubic_torch_aa_rgb(img.as_raw(), src_w, src_h, out_w, out_h).unwrap();
        assert_eq!(
            via_dynamic.to_rgb8().into_raw(),
            via_bytes.into_raw(),
            "raw-RGB torch bicubic must equal DynamicImage torch bicubic byte-for-byte"
        );
    }

    /// The passes' parity cases: up- and downscales, identity axes, extreme
    /// aspects, row lengths that leave vector tails, a large image, and the
    /// vector width pass's edges (an axis shorter than one padded window,
    /// windows of eight, nine and seven taps, a window of one tap past a
    /// multiple of eight, a downscale by 154 whose padded windows run
    /// hundreds of source pixels).
    const PIL_PASS_CASES: [(usize, usize, usize, usize); 19] = [
        (37, 23, 64, 32),
        (259, 194, 256, 192),
        (640, 480, 320, 224),
        (100, 20, 32, 32),
        (20, 100, 32, 32),
        (113, 97, 112, 96),
        (7, 5, 33, 17),
        (33, 17, 7, 5),
        (31, 23, 31, 17),
        (31, 23, 19, 23),
        (1, 1, 5, 3),
        (5, 3, 1, 1),
        (1296, 972, 1036, 777),
        (64, 5, 8, 5),
        (64, 5, 9, 5),
        (64, 5, 7, 5),
        (5, 3, 40, 3),
        (300, 4, 97, 4),
        (2000, 3, 13, 3),
    ];

    /// `w` by `h` four-channel bytes that vary with the position and `seed`.
    fn gradient_rgba(w: usize, h: usize, seed: usize) -> Vec<u8> {
        gradient_rgb(w, h, seed)
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, &[r, g, b])| [r, g, b, ((seed + i * 37) % 256) as u8])
            .collect()
    }

    /// Every byte of the 32-bit passes, three and four channels, bicubic and
    /// Lanczos, over [`PIL_PASS_CASES`], equals the per-sample `i64`
    /// computation.
    #[test]
    fn pil_passes_equal_the_per_sample_i64_reference() {
        fn check<const C: usize>(
            data: &[u8],
            (in_w, in_h, out_w, out_h): (usize, usize, usize, usize),
            filter: PilResizeFilter,
            what: &str,
        ) {
            let actual = resize_pil_bytes::<C>(
                data,
                in_w as u32,
                in_h as u32,
                out_w as u32,
                out_h as u32,
                filter,
                KernelQuantization::Pillow,
            );
            let expected = per_sample_reference_channels::<C>(
                data,
                in_w,
                in_h,
                out_w,
                out_h,
                filter,
                KernelQuantization::Pillow,
            );
            assert_same_bytes(&actual, &expected, &format!("{C} channels, {what}"));
        }
        for (f, filter) in [PilResizeFilter::Bicubic, PilResizeFilter::Lanczos]
            .into_iter()
            .enumerate()
        {
            for (seed, &case) in PIL_PASS_CASES.iter().enumerate() {
                let (in_w, in_h, out_w, out_h) = case;
                let seed = seed * 13 + f * 101;
                let what = format!("filter {f}: {in_w}x{in_h} -> {out_w}x{out_h}");
                check::<3>(&gradient_rgb(in_w, in_h, seed), case, filter, &what);
                check::<4>(&gradient_rgba(in_w, in_h, seed), case, filter, &what);
            }
        }
    }

    /// The portable and the detected arms of the two passes write the same
    /// bytes on every case, three and four channels (on a processor without
    /// the vector unit both arms are the portable loop).
    #[test]
    fn pil_pass_arms_agree() {
        fn check<const C: usize>(
            data: &[u8],
            (in_w, in_h, out_w, out_h): (usize, usize, usize, usize),
            filter: PilResizeFilter,
            what: &str,
        ) {
            let kernel = FixedKernel::new(in_w, out_w, filter, KernelQuantization::Pillow);
            let mut portable = vec![0_u8; in_h * out_w * C];
            WidthPass::Portable.rows::<C>(data, &kernel, in_w, out_w, 0, &mut portable);
            let mut detected = vec![0_u8; in_h * out_w * C];
            WidthPass::new(&kernel, in_w).rows::<C>(data, &kernel, in_w, out_w, 0, &mut detected);
            assert_same_bytes(
                &detected,
                &portable,
                &format!("width pass, {C} channels, {what}"),
            );
            let kernel = FixedKernel::new(in_h, out_h, filter, KernelQuantization::Pillow);
            let mut portable_rows = vec![0_u8; out_h * out_w * C];
            pil_v_band(&portable, &kernel, out_w * C, 0, &mut portable_rows);
            let mut detected_rows = vec![0_u8; out_h * out_w * C];
            pil_v_rows(&portable, &kernel, out_w * C, 0, &mut detected_rows);
            assert_same_bytes(
                &detected_rows,
                &portable_rows,
                &format!("height pass, {C} channels, {what}"),
            );
        }
        for (f, filter) in [PilResizeFilter::Bicubic, PilResizeFilter::Lanczos]
            .into_iter()
            .enumerate()
        {
            for (seed, &case) in PIL_PASS_CASES.iter().enumerate() {
                let (in_w, in_h, out_w, out_h) = case;
                let what = format!("filter {f}: {in_w}x{in_h} -> {out_w}x{out_h}");
                check::<3>(&gradient_rgb(in_w, in_h, seed), case, filter, &what);
                check::<4>(&gradient_rgba(in_w, in_h, seed), case, filter, &what);
            }
        }
    }

    /// The bound the 32-bit accumulators rest on, checked over a sweep of
    /// axis sizes for both filters: the rounding term plus the largest
    /// possible pixel contribution (`255 · Σ|tap|`) of any output index stays
    /// inside `i32`.
    #[test]
    fn pil_kernel_sums_fit_32_bits() {
        let sizes = [1usize, 2, 3, 5, 7, 16, 100, 1000, 4096];
        for filter in [PilResizeFilter::Bicubic, PilResizeFilter::Lanczos] {
            for &in_size in &sizes {
                for &out_size in &sizes {
                    let k = FixedKernel::new(in_size, out_size, filter, KernelQuantization::Pillow);
                    let half = 1_i64 << (k.precision - 1);
                    let worst = k
                        .taps
                        .iter()
                        .map(|taps| taps.iter().map(|&tap| i64::from(tap).abs()).sum::<i64>())
                        .max()
                        .unwrap_or(0);
                    assert!(
                        half + 255 * worst <= i64::from(i32::MAX),
                        "{in_size} -> {out_size}: Σ|tap| {worst} overflows the 32-bit sum"
                    );
                }
            }
        }
    }

    /// The bytes-level pad is the paste `image::imageops::overlay` makes of
    /// the same contain fit onto the same canvas (the pad's earlier
    /// implementation), on a padded height, a padded width and an exact fit.
    #[test]
    fn pad_to_size_pil_rgb_pastes_like_overlay() {
        for (w, h, out_w, out_h) in [
            (64u32, 48u32, 630u32, 476u32),
            (48, 64, 476, 630),
            (100, 100, 546, 546),
        ] {
            let data = gradient_rgb(w as usize, h as usize, 5);
            let color = Rgb([127, 127, 127]);
            let actual = pad_to_size_pil_rgb(&data, w, h, out_w, out_h, color).unwrap();

            // ImageOps.contain, as the reference computes it.
            let (mut target_w, mut target_h) = (out_w, out_h);
            let im_ratio = f64::from(w) / f64::from(h);
            let dest_ratio = f64::from(out_w) / f64::from(out_h);
            if im_ratio > dest_ratio {
                target_h =
                    (f64::from(h) / f64::from(w) * f64::from(out_w)).round_ties_even() as u32;
            } else if im_ratio < dest_ratio {
                target_w =
                    (f64::from(w) / f64::from(h) * f64::from(out_h)).round_ties_even() as u32;
            }
            let resized = resize_bicubic_pil_rgb(&data, w, h, target_w, target_h).unwrap();
            let mut expected = RgbImage::from_pixel(out_w, out_h, color);
            let x = (f64::from(out_w - target_w) * 0.5).round_ties_even() as i64;
            let y = (f64::from(out_h - target_h) * 0.5).round_ties_even() as i64;
            image::imageops::overlay(&mut expected, &resized, x, y);
            assert_eq!(
                (target_w, target_h) == (out_w, out_h),
                w == h,
                "{w}x{h}: the exact fit is the square case"
            );
            assert_eq!(
                actual.as_raw(),
                expected.as_raw(),
                "{w}x{h} -> {out_w}x{out_h}"
            );
        }
    }

    /// The raw-RGB video resizer must be byte-for-byte identical to the
    /// DynamicImage PIL-bicubic resizer used for images. Guards the video resize
    /// path used by `preprocess_video_rgb`.
    #[test]
    fn resize_bicubic_pil_rgb_matches_dynamic_path() {
        let (src_w, src_h) = (37u32, 23u32); // non-aligned source, non-trivial ratios
        let (out_w, out_h) = (16u32, 28u32); // downscale width, upscale height
        let mut img = RgbImage::new(src_w, src_h);
        for y in 0..src_h {
            for x in 0..src_w {
                img.put_pixel(
                    x,
                    y,
                    Rgb([
                        ((x * 7) ^ (y * 13)) as u8,
                        (x * 3 + y * 5) as u8,
                        (x + y * y) as u8,
                    ]),
                );
            }
        }
        let via_dynamic = resize_bicubic_pil(&DynamicImage::ImageRgb8(img.clone()), out_w, out_h);
        let via_bytes = resize_bicubic_pil_rgb(img.as_raw(), src_w, src_h, out_w, out_h).unwrap();
        assert_eq!(
            via_dynamic.to_rgb8().into_raw(),
            via_bytes.into_raw(),
            "raw-RGB PIL bicubic must equal DynamicImage PIL bicubic byte-for-byte"
        );
    }

    #[test]
    fn resize_bicubic_pil_rgb_skips_identity_axes_bit_exactly() {
        let (src_w, src_h) = (31u32, 23u32);
        let mut data = vec![0u8; src_w as usize * src_h as usize * 3];
        for (index, value) in data.iter_mut().enumerate() {
            *value = (index as u8).wrapping_mul(37).wrapping_add(11);
        }

        for (out_w, out_h) in [(src_w, 17), (19, src_h), (src_w, src_h)] {
            let horizontal_kernel = FixedKernel::new(
                src_w as usize,
                out_w as usize,
                PilResizeFilter::Bicubic,
                KernelQuantization::Pillow,
            );
            let horizontal = pil_resample_horizontal::<3>(
                &data,
                src_h as usize,
                src_w as usize,
                out_w as usize,
                &horizontal_kernel,
            );
            let vertical_kernel = FixedKernel::new(
                src_h as usize,
                out_h as usize,
                PilResizeFilter::Bicubic,
                KernelQuantization::Pillow,
            );
            let expected = pil_resample_vertical::<3>(
                &horizontal,
                out_h as usize,
                out_w as usize,
                &vertical_kernel,
            );
            let actual = resize_bicubic_pil_rgb(&data, src_w, src_h, out_w, out_h)
                .unwrap()
                .into_raw();

            assert_eq!(actual, expected, "identity-axis fast path changed pixels");
        }
    }

    /// `resize_bicubic_pil_rgb` rejects a buffer whose length doesn't match the
    /// declared dimensions rather than reading out of bounds.
    #[test]
    fn resize_bicubic_pil_rgb_rejects_an_empty_target() {
        let data = vec![0u8; 2 * 2 * 3];
        for (out_w, out_h) in [(0, 480), (480, 0)] {
            assert!(matches!(
                resize_bicubic_pil_rgb(&data, 2, 2, out_w, out_h),
                Err(TransformError::ShapeError(_))
            ));
        }
    }

    #[test]
    fn resize_bicubic_pil_rgb_rejects_wrong_length() {
        assert!(
            resize_bicubic_pil_rgb(&[0u8; 10], 4, 4, 2, 2).is_err(),
            "wrong-length RGB buffer must error, not panic"
        );
    }

    /// The two quantizations share the kernel; only the tap scale differs:
    /// Pillow's fixed 22 bits against PyTorch's `uint8` rule, 16-bit taps at
    /// the finest scale under 2^15. Bicubic weights never exceed 2: the rule
    /// lands at 14 bits for an upscale (centre tap near 1), one more for a
    /// downscale whose centre tap is below 0.5.
    #[test]
    fn torch_uint8_quantization_scales_taps_to_int16() {
        for (in_size, out_size) in [(512, 768), (3000, 960), (64, 768), (7, 5)] {
            let pillow = FixedKernel::new(
                in_size,
                out_size,
                PilResizeFilter::Bicubic,
                KernelQuantization::Pillow,
            );
            let torch = FixedKernel::new(
                in_size,
                out_size,
                PilResizeFilter::Bicubic,
                KernelQuantization::TorchUint8,
            );
            assert_eq!(pillow.precision, PIL_PRECISION_BITS);
            assert_eq!(pillow.bounds, torch.bounds, "{in_size}->{out_size}");
            assert!(
                (1..PIL_PRECISION_BITS).contains(&torch.precision),
                "{in_size}->{out_size}: precision {}",
                torch.precision
            );
            let max_tap = torch.taps.iter().flatten().copied().max().unwrap_or(0);
            assert!(
                ((1 << 14)..(1 << 15)).contains(&max_tap),
                "{in_size}->{out_size}: max tap {max_tap} at {} bits",
                torch.precision
            );
            assert!(torch
                .taps
                .iter()
                .flatten()
                .all(|&tap| i16::try_from(tap).is_ok()));
            // Each output's taps sum to one in the quantization's own scale,
            // give or take the rounding of each tap.
            for taps in &torch.taps {
                let sum: i64 = taps.iter().map(|&tap| i64::from(tap)).sum();
                let one = 1_i64 << torch.precision;
                assert!((sum - one).abs() <= taps.len() as i64, "{sum} vs {one}");
            }
        }
    }

    #[test]
    fn resize_bicubic_torch_aa_rgb_identity_and_errors() {
        let image = create_test_image(6, 4, Rgb([10, 20, 30]));
        let raw = image.to_rgb8().into_raw();
        let same = resize_bicubic_torch_aa_rgb(&raw, 6, 4, 6, 4).unwrap();
        assert_eq!(same.as_raw(), &raw);
        // A flat image stays flat through both passes at any size.
        let resized = resize_bicubic_torch_aa_rgb(&raw, 6, 4, 9, 7).unwrap();
        assert_eq!(resized.dimensions(), (9, 7));
        assert!(resized.pixels().all(|pixel| *pixel == Rgb([10, 20, 30])));
        assert!(resize_bicubic_torch_aa_rgb(&raw, 6, 4, 0, 4).is_err());
        assert!(resize_bicubic_torch_aa_rgb(&raw[..10], 6, 4, 3, 2).is_err());
    }

    #[test]
    fn test_to_tensor_shape() {
        let img = create_test_image(10, 20, Rgb([255, 128, 0]));
        let tensor = to_tensor(&img);
        assert_eq!(tensor.shape(), &[3, 20, 10]); // [C, H, W]
    }

    #[test]
    fn test_to_tensor_values() {
        let img = create_test_image(2, 2, Rgb([255, 128, 0]));
        let tensor = to_tensor(&img);

        // Check normalization to [0, 1]
        assert!((tensor[[0, 0, 0]] - 1.0).abs() < 1e-6); // R=255 -> 1.0
        assert!((tensor[[1, 0, 0]] - 0.502).abs() < 0.01); // G=128 -> ~0.5
        assert!((tensor[[2, 0, 0]] - 0.0).abs() < 1e-6); // B=0 -> 0.0
    }

    #[test]
    fn test_to_tensor_no_norm() {
        let img = create_test_image(2, 2, Rgb([255, 128, 64]));
        let tensor = to_tensor_no_norm(&img);

        assert!((tensor[[0, 0, 0]] - 255.0).abs() < 1e-6);
        assert!((tensor[[1, 0, 0]] - 128.0).abs() < 1e-6);
        assert!((tensor[[2, 0, 0]] - 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut tensor = Array3::<f32>::from_elem((3, 2, 2), 0.5);
        let mean = [0.5, 0.5, 0.5];
        let std = [0.5, 0.5, 0.5];

        normalize(&mut tensor, &mean, &std);

        // (0.5 - 0.5) / 0.5 = 0.0
        for val in &tensor {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_rescale() {
        let mut tensor = Array3::<f32>::from_elem((3, 2, 2), 255.0);
        rescale(&mut tensor, 1.0 / 255.0);

        for val in &tensor {
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_resize() {
        let img = create_test_image(100, 50, Rgb([128, 128, 128]));
        let resized = resize(&img, 50, 25, FilterType::Triangle);

        assert_eq!(resized.width(), 50);
        assert_eq!(resized.height(), 25);
    }

    #[test]
    fn test_resize_rgb_bytes_matches_resize() {
        let mut rgb = RgbImage::new(3, 2);
        for y in 0..2 {
            for x in 0..3 {
                rgb.put_pixel(x, y, Rgb([(x * 40) as u8, (y * 90) as u8, 128]));
            }
        }
        let img = DynamicImage::ImageRgb8(rgb.clone());
        let expected = resize(&img, 2, 2, FilterType::Triangle).to_rgb8();
        let actual = resize_rgb_bytes(rgb.as_raw(), 3, 2, 2, 2, FilterType::Triangle)
            .expect("resize_rgb_bytes should resize valid RGB input");

        assert_eq!(actual.as_raw(), expected.as_raw());
    }

    #[test]
    fn test_resize_rgb_bytes_rejects_invalid_length() {
        let result = resize_rgb_bytes(&[1, 2, 3, 4, 5], 2, 1, 1, 1, FilterType::Triangle);
        assert!(matches!(result, Err(TransformError::ShapeError(_))));
    }

    #[test]
    fn test_center_crop() {
        let img = create_test_image(100, 100, Rgb([128, 128, 128]));
        let cropped = center_crop(&img, 50, 50);

        assert_eq!(cropped.width(), 50);
        assert_eq!(cropped.height(), 50);
    }

    #[test]
    fn center_crop_is_the_generic_crop() {
        let (w, h) = (97u32, 61u32);
        let image = DynamicImage::ImageRgb8(
            RgbImage::from_raw(w, h, gradient_rgb(w as usize, h as usize, 13))
                .expect("gradient size"),
        );
        // A crop inside the image, one flush with an edge, one wider than the
        // image on one axis (clipped there, centred on the other).
        for (crop_w, crop_h) in [(50, 50), (97, 20), (20, 61), (200, 10)] {
            let expected = image.crop_imm(
                w.saturating_sub(crop_w) / 2,
                h.saturating_sub(crop_h) / 2,
                crop_w.min(w),
                crop_h.min(h),
            );
            let cropped = center_crop(&image, crop_w, crop_h);
            assert_eq!(
                cropped.dimensions(),
                expected.dimensions(),
                "{crop_w}x{crop_h}"
            );
            assert_eq!(cropped.as_bytes(), expected.as_bytes(), "{crop_w}x{crop_h}");
        }
    }

    #[test]
    fn paste_rgb_is_the_generic_paste() {
        // A paste offset on either axis, a flush paste, an image the canvas
        // clips, and one placed past the canvas; on black and on a colour.
        for (image_w, image_h, canvas_w, canvas_h, x, y) in [
            (20u32, 10u32, 24u32, 24u32, 2u32, 7u32),
            (10, 20, 24, 24, 7, 2),
            (24, 7, 24, 24, 0, 8),
            (24, 24, 24, 24, 0, 0),
            (30, 10, 24, 24, 3, 20),
            (5, 5, 24, 24, 30, 1),
        ] {
            for background in [[0, 0, 0], [127, 10, 250]] {
                let image = RgbImage::from_raw(
                    image_w,
                    image_h,
                    gradient_rgb(image_w as usize, image_h as usize, 3),
                )
                .expect("gradient size");
                let mut expected = RgbImage::from_pixel(canvas_w, canvas_h, Rgb(background));
                image::imageops::overlay(&mut expected, &image, i64::from(x), i64::from(y));
                let pasted = paste_rgb(&image, canvas_w, canvas_h, x, y, background);
                assert_eq!(
                    pasted.as_raw(),
                    expected.as_raw(),
                    "{image_w}x{image_h} at ({x}, {y}) on {canvas_w}x{canvas_h} {background:?}"
                );
            }
        }
    }

    #[test]
    fn center_crop_and_paste_rgb_stay_total_on_a_zero_width_image() {
        // The generic crop and paste give an empty crop and an untouched
        // canvas for a zero-width image; the row copies do the same.
        let empty = RgbImage::new(0, 61);
        let image = DynamicImage::ImageRgb8(empty.clone());
        for (crop_w, crop_h) in [(50, 20), (0, 61), (200, 10)] {
            let expected = image.crop_imm(0, 61u32.saturating_sub(crop_h) / 2, 0, crop_h.min(61));
            let cropped = center_crop(&image, crop_w, crop_h);
            assert_eq!(
                cropped.dimensions(),
                expected.dimensions(),
                "{crop_w}x{crop_h}"
            );
            assert!(cropped.as_bytes().is_empty(), "{crop_w}x{crop_h}");
        }
        let expected = RgbImage::from_pixel(24, 24, Rgb([127, 10, 250]));
        assert_eq!(
            paste_rgb(&empty, 24, 24, 3, 2, [127, 10, 250]).as_raw(),
            expected.as_raw()
        );
    }

    #[test]
    fn test_expand_to_square_horizontal() {
        let img = create_test_image(100, 50, Rgb([255, 0, 0]));
        let background = Rgb([0, 0, 0]);
        let squared = expand_to_square(&img, background);

        assert_eq!(squared.width(), 100);
        assert_eq!(squared.height(), 100);
    }

    #[test]
    fn test_expand_to_square_vertical() {
        let img = create_test_image(50, 100, Rgb([255, 0, 0]));
        let background = Rgb([0, 0, 0]);
        let squared = expand_to_square(&img, background);

        assert_eq!(squared.width(), 100);
        assert_eq!(squared.height(), 100);
    }

    #[test]
    fn test_expand_to_square_already_square() {
        let img = create_test_image(100, 100, Rgb([255, 0, 0]));
        let background = Rgb([0, 0, 0]);
        let squared = expand_to_square(&img, background);

        assert_eq!(squared.width(), 100);
        assert_eq!(squared.height(), 100);
    }

    #[test]
    fn test_stack_batch() {
        let t1 = Array3::<f32>::zeros((3, 10, 10));
        let t2 = Array3::<f32>::ones((3, 10, 10));

        let batch = stack_batch(&[t1, t2]).unwrap();

        assert_eq!(batch.shape(), &[2, 3, 10, 10]);
    }

    #[test]
    fn stack_batch_holds_the_tensors_in_order() {
        let first =
            Array3::<f32>::from_shape_fn((3, 4, 5), |(c, y, x)| (c * 100 + y * 10 + x) as f32);
        let second = first.mapv(|v| -v - 1.0);
        // A tensor that is not in standard layout (its axes reversed) reads
        // element by element.
        let third =
            Array3::<f32>::from_shape_fn((5, 4, 3), |(x, y, c)| (c * 1000 + y * 10 + x) as f32)
                .reversed_axes();
        assert!(third.as_slice().is_none());
        let tensors = [first, second, third];

        let batch = stack_batch(&tensors).unwrap();

        // The zeroed form: a zero batch with every tensor assigned.
        let mut expected = Array4::<f32>::zeros((3, 3, 4, 5));
        for (i, tensor) in tensors.iter().enumerate() {
            expected.slice_mut(s![i, .., .., ..]).assign(tensor);
        }
        assert_eq!(batch, expected);
    }

    #[test]
    fn test_stack_batch_empty() {
        let result = stack_batch(&[]);
        assert!(matches!(result, Err(TransformError::EmptyBatch)));
    }

    #[test]
    fn test_pil_to_filter() {
        assert!(matches!(pil_to_filter(Some(0)), FilterType::Nearest));
        assert!(matches!(pil_to_filter(Some(1)), FilterType::Lanczos3));
        assert!(matches!(pil_to_filter(Some(2)), FilterType::Triangle));
        assert!(matches!(pil_to_filter(Some(3)), FilterType::CatmullRom));
        assert!(matches!(pil_to_filter(None), FilterType::Triangle));
    }

    #[test]
    fn test_mean_to_rgb() {
        let mean = [0.5, 0.25, 1.0];
        let rgb = mean_to_rgb(&mean);

        assert_eq!(rgb[0], 128);
        assert_eq!(rgb[1], 64);
        assert_eq!(rgb[2], 255);
    }

    fn chessboard(square_size: u32, on_top_left: bool) -> TransparentBgConfig {
        TransparentBgConfig {
            pattern: TransparentBgPattern::Chessboard,
            chessboard_square_size: square_size,
            chessboard_square_on_top_left: on_top_left,
            chessboard_white_value: 255,
            chessboard_gray_value: 180,
        }
    }

    /// Mirrors the reference `_create_chessboard_background`:
    /// `bg[y, x] = gray if (y//s + x//s) % 2 == (1 if on_top_left else 0)`.
    fn reference_board(config: TransparentBgConfig, x: u32, y: u32) -> u8 {
        let s = config.chessboard_square_size;
        let gray_cell = u32::from(config.chessboard_square_on_top_left);
        if (y / s + x / s) % 2 == gray_cell {
            config.chessboard_gray_value
        } else {
            config.chessboard_white_value
        }
    }

    #[test]
    fn chessboard_background_matches_reference() {
        for on_top_left in [true, false] {
            let config = chessboard(8, on_top_left);
            let transparent =
                DynamicImage::from(image::RgbaImage::from_pixel(24, 24, Rgba([0, 0, 0, 0])));
            let out = fill_transparent_bg(&transparent, config);
            for y in 0..24 {
                for x in 0..24 {
                    let expected = reference_board(config, x, y);
                    assert_eq!(
                        out.get_pixel(x, y),
                        &Rgb([expected, expected, expected]),
                        "({x},{y}) with on_top_left={on_top_left}"
                    );
                }
            }
        }
    }

    #[test]
    fn fill_transparent_bg_blends_partial_alpha() {
        // The reference computes `a*img + (1-a)*bg` in float32, then truncates
        // via `.astype(np.uint8)`. Check that arithmetic exactly.
        let mut img = image::RgbaImage::new(2, 1);
        img.put_pixel(0, 0, Rgba([200, 100, 50, 64]));
        img.put_pixel(1, 0, Rgba([200, 100, 50, 192]));
        let config = TransparentBgConfig {
            pattern: TransparentBgPattern::Gray, // flat 128, so no board phase
            ..Default::default()
        };

        let out = fill_transparent_bg(&DynamicImage::from(img), config);
        for (x, alpha) in [(0u32, 64.0f32), (1, 192.0)] {
            let a = alpha / 255.0;
            for (c, src) in [200.0f32, 100.0, 50.0].into_iter().enumerate() {
                let expected = (a * src + (1.0 - a) * 128.0) as u8;
                assert_eq!(out.get_pixel(x, 0)[c], expected, "x={x} channel={c}");
            }
        }
    }

    #[test]
    fn fill_transparent_bg_is_identity_without_alpha() {
        let rgb = create_test_image(4, 4, Rgb([12, 34, 56]));
        let out = fill_transparent_bg(&rgb, chessboard(2, true));
        assert!(out.pixels().all(|p| p == &Rgb([12, 34, 56])));
    }

    #[test]
    fn fill_transparent_bg_survives_zero_square_size() {
        // The reference raises on square_size=0; clamp instead of dividing by zero.
        let transparent =
            DynamicImage::from(image::RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0])));
        let out = fill_transparent_bg(&transparent, chessboard(0, true));
        assert_eq!(out.dimensions(), (4, 4));
    }

    #[test]
    fn transparent_bg_config_fills_missing_fields() {
        // A model may ship only `pattern`; the rest must fall back rather than
        // failing the whole preprocessor_config parse.
        let config: TransparentBgConfig =
            serde_json::from_str(r#"{"pattern": "chessboard"}"#).unwrap();
        assert_eq!(config.pattern, TransparentBgPattern::Chessboard);
        assert_eq!(
            config.chessboard_square_size,
            TransparentBgConfig::default().chessboard_square_size
        );
    }

    #[test]
    fn paint_rgb_onto_planes_maps_the_image_and_fills_the_margin_with_bias() {
        // A 2x2 image on a 3x2 canvas: the right column is margin.
        let rgb: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let scale = [0.5, 0.25, 2.0];
        let bias = [-1.0, 0.5, 3.0];
        let mut planes = vec![f32::NAN; 3 * 3 * 2];
        paint_rgb_onto_planes(&rgb, (2, 2), (3, 2), scale, bias, &mut planes);
        for c in 0..3 {
            for y in 0..2 {
                for x in 0..3 {
                    let expected = if x < 2 {
                        f32::from(rgb[(y * 2 + x) * 3 + c]) * scale[c] + bias[c]
                    } else {
                        bias[c]
                    };
                    assert_eq!(
                        planes[c * 6 + y * 3 + x].to_bits(),
                        expected.to_bits(),
                        "channel {c} at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn paint_rgb_onto_planes_by_table_maps_through_the_table_and_fills_the_margin_with_its_zero() {
        // A 2x2 image on a 3x2 canvas: the right column is margin. The table
        // tells the channels and the levels apart, and no multiply-add
        // produces it.
        let rgb: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let table: [[f32; 256]; 3] = std::array::from_fn(|c| {
            std::array::from_fn(|level| {
                let sign = if level % 2 == 0 { 1.0 } else { -1.0 };
                (level as f32 * 0.5 + 1.0) * sign + (c as f32 + 1.0) * 1000.0
            })
        });
        let mut planes = vec![f32::NAN; 3 * 3 * 2];
        paint_rgb_onto_planes_by_table(&rgb, (2, 2), (3, 2), &table, &mut planes);
        for c in 0..3 {
            for y in 0..2 {
                for x in 0..3 {
                    let level = if x < 2 { rgb[(y * 2 + x) * 3 + c] } else { 0 };
                    assert_eq!(
                        planes[c * 6 + y * 3 + x].to_bits(),
                        table[c][usize::from(level)].to_bits(),
                        "channel {c} at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn deinterleave_by_table_is_bit_identical_banded_and_serial() {
        // A row wide enough for the banded path; the serial kernel is the
        // oracle.
        let pixels = 1 << 20;
        let rgb: Vec<u8> = (0..pixels * 3).map(|i| (i % 251) as u8).collect();
        let table: [[f32; 256]; 3] =
            std::array::from_fn(|c| std::array::from_fn(|level| (level * 3 + c) as f32 / 7.0));
        let mut serial = vec![0.0_f32; 3 * pixels];
        {
            let (r, rest) = serial.split_at_mut(pixels);
            let (g, b) = rest.split_at_mut(pixels);
            deinterleave_contiguous_by_table(&rgb, r, g, b, &table);
        }
        let mut banded = vec![0.0_f32; 3 * pixels];
        {
            let (r, rest) = banded.split_at_mut(pixels);
            let (g, b) = rest.split_at_mut(pixels);
            deinterleave_rgb_to_planes_by_table(&rgb, r, g, b, &table);
        }
        assert!(serial
            .iter()
            .zip(&banded)
            .all(|(s, b)| s.to_bits() == b.to_bits()));
        assert_eq!(serial[0].to_bits(), table[0][0].to_bits());
        assert_eq!(serial[pixels].to_bits(), table[1][1].to_bits());
    }

    #[test]
    fn paint_rgb_onto_planes_crops_an_image_larger_than_the_canvas() {
        // A 3x3 image on a 2x2 canvas keeps its top-left 2x2 pixels.
        let rgb: Vec<u8> = (0..27).collect();
        let mut planes = vec![f32::NAN; 3 * 2 * 2];
        paint_rgb_onto_planes(&rgb, (3, 3), (2, 2), [1.0; 3], [0.0; 3], &mut planes);
        for c in 0..3 {
            for y in 0..2 {
                for x in 0..2 {
                    assert_eq!(
                        planes[c * 4 + y * 2 + x].to_bits(),
                        f32::from(rgb[(y * 3 + x) * 3 + c]).to_bits(),
                        "channel {c} at ({x}, {y})"
                    );
                }
            }
        }
    }

    /// [`resize_bicubic_torch_aa_f32`] carries torch's bits: four seeded
    /// inputs (an LCG's bytes) through a four-tap upscale, a seven-tap and a
    /// seventeen-tap one-row downscale and a mixed resize, against
    /// `torch.nn.functional.interpolate(mode="bicubic", antialias=True)` on
    /// the CPU (torch 2.13.0): the digest of every output's bits and, for a
    /// readable failure, four elements per case where the previous evaluation
    /// (the cubic in double, every tap added plainly) parted by one ULP. The
    /// dispatching entry point, the generic body and, where the processor has
    /// it, the FMA body are each driven directly, so a host with the
    /// instruction covers the fallback too.
    #[test]
    fn resize_bicubic_torch_aa_f32_carries_the_reference_bits() {
        use sha2::{Digest, Sha256};

        struct Case {
            width: u32,
            height: u32,
            out_w: u32,
            out_h: u32,
            seed: u32,
            // SHA-256 of the little-endian f32 bits of the reference's output.
            digest: &'static str,
            // (index, bits) where the previous evaluation parted by one ULP.
            witnesses: [(usize, u32); 4],
        }
        fn lcg_bytes(count: usize, seed: u32) -> Vec<u8> {
            let mut state = seed;
            (0..count)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
                    ((state >> 16) & 0xff) as u8
                })
                .collect()
        }
        fn hex(bytes: &[u8]) -> String {
            bytes.iter().map(|byte| format!("{byte:02x}")).collect()
        }

        let cases = [
            Case {
                width: 5,
                height: 7,
                out_w: 3,
                out_h: 4,
                seed: 11,
                digest: "d4bfc041c7c9ab5e8f3740ddb25b93c0dfdc4e5a7187b92c32f201c6776cc240",
                witnesses: [
                    (1, 0x4336_b058),
                    (2, 0x42d1_fa32),
                    (3, 0x434e_08f2),
                    (4, 0x4313_c030),
                ],
            },
            Case {
                width: 13,
                height: 1,
                out_w: 7,
                out_h: 1,
                seed: 23,
                digest: "fb846101957b27905415b007ae9cd8daeaf9656d787e71c0d2eeb494dff0afa8",
                witnesses: [
                    (0, 0x4362_6944),
                    (1, 0x4299_0234),
                    (3, 0x4303_883b),
                    (4, 0x421e_1e2c),
                ],
            },
            Case {
                width: 33,
                height: 3,
                out_w: 8,
                out_h: 3,
                seed: 37,
                digest: "de6d9663c4ce3e6c48f5a0d77d081a97667bedc70ff3b1f243861448a3a2d453",
                witnesses: [
                    (0, 0x4302_3f58),
                    (1, 0x42ef_062d),
                    (2, 0x42dc_09e2),
                    (4, 0x430d_68a6),
                ],
            },
            Case {
                width: 11,
                height: 5,
                out_w: 12,
                out_h: 8,
                seed: 5,
                digest: "4c42196b1e0615e6780584fdbfb1e149752ff3248eb4ad4a44d8b1f618d22cca",
                witnesses: [
                    (1, 0xc0ab_e092),
                    (4, 0x434e_ddc0),
                    (5, 0x42d7_239c),
                    (6, 0x4337_999a),
                ],
            },
        ];
        for case in cases {
            let Case {
                width,
                height,
                out_w,
                out_h,
                seed,
                digest,
                witnesses,
            } = case;
            let data = lcg_bytes((width * height * 3) as usize, seed);
            let (x_bounds, x_weights) = torch_aa_bicubic_weights(width as usize, out_w as usize);
            let (y_bounds, y_weights) = torch_aa_bicubic_weights(height as usize, out_h as usize);
            let x = TorchAaAxis {
                bounds: &x_bounds,
                weights: &x_weights,
            };
            let y = TorchAaAxis {
                bounds: &y_bounds,
                weights: &y_weights,
            };
            #[cfg(target_arch = "x86_64")]
            let fma = torch_aa_passes_with_fma(&data, width as usize, &x, &y);
            #[cfg(not(target_arch = "x86_64"))]
            let fma: Option<Vec<f32>> = None;
            let bodies = [
                (
                    "the dispatching entry point",
                    Some(resize_bicubic_torch_aa_f32(&data, width, height, out_w, out_h).unwrap()),
                ),
                (
                    "the generic body",
                    Some(torch_aa_passes(&data, width as usize, &x, &y)),
                ),
                ("the FMA body", fma),
            ];
            for (body, out) in bodies
                .into_iter()
                .filter_map(|(body, out)| out.map(|out| (body, out)))
            {
                assert_eq!(out.len(), (out_w * out_h * 3) as usize);
                for (index, bits) in witnesses {
                    assert_eq!(
                        out[index].to_bits(),
                        bits,
                        "{width}x{height} -> {out_w}x{out_h} element {index} through {body}"
                    );
                }
                let bytes: Vec<u8> = out.iter().flat_map(|value| value.to_le_bytes()).collect();
                assert_eq!(
                    hex(&Sha256::digest(&bytes)),
                    digest,
                    "{width}x{height} -> {out_w}x{out_h} through {body}"
                );
            }
        }
    }
}
