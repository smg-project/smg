//! Image transformation functions for vision preprocessing.
//!
//! This module provides composable transforms that match HuggingFace image processor
//! behavior, enabling pure Rust preprocessing without Python dependencies.

use std::{cell::RefCell, f64::consts::PI, ops::Range};

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

/// [`build_planar_tensor`] through a per-channel table:
/// `output[c][i] = table[c][raw[i*3 + c]]`.
fn build_planar_tensor_by_table(
    raw: &[u8],
    w: usize,
    h: usize,
    table: &[[f32; 256]; 3],
) -> Array3<f32> {
    let pixels = h * w;
    let mut data = scratch::take_f32(3 * pixels);
    let (r_plane, rest) = data.split_at_mut(pixels);
    let (g_plane, b_plane) = rest.split_at_mut(pixels);

    deinterleave_rgb_to_planes_by_table(raw, r_plane, g_plane, b_plane, table);

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

/// [`to_tensor`] and the normalization in one pass through a per-channel
/// table of the 256 byte levels (`output[c][i] = table[c][raw[i*3 + c]]`,
/// the planes [`to_tensor_then_normalize`] builds), for a processor that
/// keeps the table of its reference's fused rescale-and-normalize
/// ([`PixelNorm::table_f32`](crate::encoder_inputs::PixelNorm::table_f32)),
/// which no single multiply-add reproduces in the last bit.
pub fn to_tensor_by_table(image: &DynamicImage, table: &[[f32; 256]; 3]) -> Array3<f32> {
    let (w, h, raw) = rgb_bytes(image);
    build_planar_tensor_by_table(&raw, w, h, table)
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

/// Convert image to tensor and normalize in the steps of torchvision's
/// `Compose([ToTensor(), Normalize(mean, std)])`: the byte divided by 255,
/// the mean subtracted, the std divided, each rounded to `f32`. The fused
/// [`to_tensor_and_normalize`] differs from these three roundings in the
/// last bit on about half of the byte values; a reference that composes the
/// two transforms takes this one.
pub fn to_tensor_then_normalize(
    image: &DynamicImage,
    mean: &[f64; 3],
    std: &[f64; 3],
) -> Array3<f32> {
    let (w, h, raw) = rgb_bytes(image);
    build_planar_tensor_by_table(&raw, w, h, &to_tensor_then_normalize_table(mean, std))
}

/// The per-channel table of [`to_tensor_then_normalize`]: every byte through
/// the three roundings of `ToTensor` then `Normalize(mean, std)`, for a
/// processor that keeps the table and reads it per pixel itself.
pub(crate) fn to_tensor_then_normalize_table(mean: &[f64; 3], std: &[f64; 3]) -> [[f32; 256]; 3] {
    std::array::from_fn(|c| {
        let (mean, std) = (mean[c] as f32, std[c] as f32);
        std::array::from_fn(|byte| (byte as f32 / 255.0 - mean) / std)
    })
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

/// A buffer of `len` bytes for a kernel that writes every one of them: a plain
/// allocation filled in place, never a zeroed allocation. The allocators
/// serve a large zeroed block by handing its pages back to the kernel and
/// taking them again page fault by page fault (jemalloc zeroes a recycled
/// block that way; glibc's zeroed allocation of a fresh heap top relies on
/// fresh pages), which costs more than the kernel's own pass on a buffer of
/// the image's scale; a plain allocation reuses the pages the allocator kept.
#[expect(
    clippy::slow_vector_initialization,
    reason = "the zero-filled `vec!` is the calloc this helper exists to avoid; see its doc comment"
)]
fn plain_buffer(len: usize) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(len);
    buffer.resize(len, 0);
    buffer
}

/// [`plain_buffer`] for a kernel that writes `f32` values.
fn plain_f32(len: usize) -> Vec<f32> {
    let mut buffer = Vec::with_capacity(len);
    buffer.resize(len, 0.0);
    buffer
}

/// The resizer's destination image over a [`plain_buffer`] rather than the
/// zeroed one its own constructor allocates; the resizer writes every pixel.
fn fir_destination(width: u32, height: u32, pixel_type: PixelType) -> FirImage<'static> {
    let len = width as usize * height as usize * pixel_type.size();
    FirImage::from_vec_u8(width, height, plain_buffer(len), pixel_type)
        .unwrap_or_else(|_| FirImage::new(width, height, pixel_type))
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
    let mut dst = fir_destination(width, height, pixel_type);
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
    let Some(mut dst) =
        RgbImage::from_raw(width, height, plain_buffer(dst_stride * height as usize))
    else {
        return image::imageops::resize(src, width, height, FilterType::CatmullRom);
    };
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
    let mut dst = fir_destination(target_width, target_height, PixelType::U8x3);
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
const PIL_BILINEAR_SUPPORT: f64 = 1.0;

#[derive(Clone, Copy)]
pub(super) enum PilResizeFilter {
    Bicubic,
    Lanczos,
    /// Pillow's BILINEAR: the triangle filter, support 1, widened by the
    /// downscale factor like the others.
    Bilinear,
}

impl PilResizeFilter {
    fn support(self) -> f64 {
        match self {
            Self::Bicubic => PIL_BICUBIC_SUPPORT,
            Self::Lanczos => PIL_LANCZOS_SUPPORT,
            Self::Bilinear => PIL_BILINEAR_SUPPORT,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Self::Bicubic => pil_cubic(x),
            Self::Lanczos => pil_lanczos(x),
            Self::Bilinear => pil_bilinear(x),
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

#[inline]
fn pil_bilinear(x: f64) -> f64 {
    // Pillow's `bilinear_filter`: the triangle.
    let x = x.abs();
    if x < PIL_BILINEAR_SUPPORT {
        1.0 - x
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

    /// The kernel's entries for the outputs `window` alone, in place: those
    /// outputs keep the taps and the source bounds they have in the whole, so
    /// a pass run on this kernel writes the whole's bytes there.
    fn window(mut self, window: Range<usize>) -> Self {
        self.bounds.truncate(window.end);
        self.bounds.drain(..window.start);
        self.taps.truncate(window.end);
        self.taps.drain(..window.start);
        self
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
/// loop; on an x86-64 processor that reports AVX2, the 16-bit multiply-add
/// steps of [`avx2::h_band`] over the kernel's taps split for them; on one
/// that reports AVX-512 VNNI and VBMI, the byte dot products of
/// [`avx512::h_band`] over the taps' byte chunks, for an axis whose output
/// groups fit its gather ([`GroupTaps`]).
enum WidthPass {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2(SplitTaps),
    #[cfg(target_arch = "x86_64")]
    Avx512(GroupTaps),
}

/// The vector width kernels' per-row scratch (the AVX2 arm's spread rows,
/// the AVX-512 arm's row tails and 32-bit sums), owned by the pass's caller:
/// a band loop sizes it once and every band reuses it, where a buffer per
/// band was a zeroed allocation per band (and, past the sizes the allocator
/// keeps cached pages for, one faulted in page by page: about 280 of them per
/// 8k-class image). The kernels write the bytes their rows cover and zero the
/// padding they read past them at the top of every band, so the scratch only
/// keeps the allocation: nothing a kernel reads depends on what it held.
#[derive(Default)]
#[cfg_attr(
    not(target_arch = "x86_64"),
    expect(dead_code, reason = "the portable width kernel keeps no scratch")
)]
struct BandScratch {
    bytes: Vec<u8>,
    sums: Vec<i32>,
}

#[cfg(target_arch = "x86_64")]
impl BandScratch {
    /// `len` bytes, zeroed when the length is new to this scratch.
    fn bytes(&mut self, len: usize) -> &mut [u8] {
        Self::sized(&mut self.bytes, len)
    }

    /// `bytes` bytes and `sums` 32-bit lanes, each zeroed when its length is
    /// new to this scratch.
    fn parts(&mut self, bytes: usize, sums: usize) -> (&mut [u8], &mut [i32]) {
        (
            Self::sized(&mut self.bytes, bytes),
            Self::sized(&mut self.sums, sums),
        )
    }

    fn sized<T: Clone + Default>(buf: &mut Vec<T>, len: usize) -> &mut [T] {
        if buf.len() != len {
            buf.clear();
            buf.resize(len, T::default());
        }
        buf
    }
}

impl WidthPass {
    #[cfg(target_arch = "x86_64")]
    fn new<const C: usize>(kernel: &FixedKernel) -> Self {
        if avx512::detected() {
            if let Some(groups) = GroupTaps::new::<C>(kernel) {
                return Self::Avx512(groups);
            }
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            Self::Avx2(SplitTaps::new(kernel, 0, kernel.bounds.len(), 4))
        } else {
            Self::Portable
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn new<const C: usize>(_kernel: &FixedKernel) -> Self {
        Self::Portable
    }

    /// Output rows `[y0, y0 + band.len() / (out_w * C))` of the width pass;
    /// a vector arm detects its features and calls in one place, as the
    /// crate's NEON kernels and the FMA pass keep them, and works in the
    /// caller's `scratch` (one per band loop, reused band after band).
    #[cfg_attr(
        target_arch = "x86_64",
        expect(
            unsafe_code,
            reason = "the target-feature function runs only after this function detected its features"
        )
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "the pass's scratch rides along so every band of a pass reuses it"
    )]
    fn rows<const C: usize>(
        &self,
        src: &[u8],
        kernel: &FixedKernel,
        in_w: usize,
        out_w: usize,
        y0: usize,
        band: &mut [u8],
        #[cfg_attr(
            not(target_arch = "x86_64"),
            expect(
                unused_variables,
                reason = "the portable width kernel keeps no scratch"
            )
        )]
        scratch: &mut BandScratch,
    ) {
        match self {
            Self::Portable => pil_h_band::<C>(src, kernel, in_w, out_w, y0, band),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(split) => {
                if !std::arch::is_x86_feature_detected!("avx2") {
                    return pil_h_band::<C>(src, kernel, in_w, out_w, y0, band);
                }
                // SAFETY: `is_x86_feature_detected!("avx2")` answered true just
                // above, the only requirement of a `target_feature(enable =
                // "avx2")` function.
                unsafe { avx2::h_band::<C>(src, kernel, split, in_w, out_w, y0, band, scratch) }
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512(groups) => {
                if !avx512::detected() {
                    return pil_h_band::<C>(src, kernel, in_w, out_w, y0, band);
                }
                // SAFETY: `avx512::detected()` answered true just above: the
                // processor reports every feature the `target_feature`
                // function enables, its only requirement.
                unsafe { avx512::h_band::<C>(src, kernel, groups, in_w, out_w, y0, band, scratch) }
            }
        }
    }
}

/// The width of a tap's low half in the vector arms' split: a tap `t` is
/// taken as `hi = (t + 2^10) >> 11`, its nearest multiple of 2^11, and
/// `lo = t - (hi << 11)` in `[-2^10, 2^10)`, so that `t = hi · 2^11 + lo`
/// exactly and `Σ pixel · t = (Σ pixel · hi) << 11 + Σ pixel · lo`. A tap is
/// below 2^23 in magnitude ([`FixedKernel`]), so both halves fit a 16-bit
/// lane; the rounded split keeps a small negative tap's low half small (a
/// floor split would give it `2^11 - |t|`), so either partial sum fits `i32`
/// over the longest window of either filter
/// (`pil_split_taps_fit_the_16_bit_lanes_and_32_bit_sums`).
#[cfg(any(target_arch = "x86_64", test))]
const SPLIT_LOW_BITS: u32 = 11;

/// `(hi, lo)` of a tap, see [`SPLIT_LOW_BITS`].
#[cfg(any(target_arch = "x86_64", test))]
fn split_tap(tap: i32) -> (i16, i16) {
    let hi = (tap + (1 << (SPLIT_LOW_BITS - 1))) >> SPLIT_LOW_BITS;
    // Below 2^12 in magnitude by the kernel's bound; a tap outside it would
    // be a broken invariant, not a value to clamp.
    debug_assert!(i16::try_from(hi).is_ok(), "tap {tap} exceeds the split");
    (hi as i16, (tap - (hi << SPLIT_LOW_BITS)) as i16)
}

/// The taps of `count` outputs from `first` of one axis as the vector arms
/// read them: every output's taps zero-padded to the longest window rounded
/// up to a multiple of `step`, split into their halves, and the halves of
/// consecutive taps `(t0, t1)`, `(t2, t3)`, ... packed two 16-bit lanes to an
/// `i32`, the pair a broadcast spreads for the multiply-add of adjacent
/// lanes; `pairs` of them per output at `hi[x * pairs..]` and `lo[x *
/// pairs..]`.
#[cfg(target_arch = "x86_64")]
struct SplitTaps {
    hi: Vec<i32>,
    lo: Vec<i32>,
    pairs: usize,
}

#[cfg(target_arch = "x86_64")]
impl SplitTaps {
    fn new(kernel: &FixedKernel, first: usize, count: usize, step: usize) -> Self {
        let outputs = &kernel.taps[first..first + count];
        let longest = outputs.iter().map(Vec::len).max().unwrap_or(0);
        let pairs = longest.max(1).next_multiple_of(step) / 2;
        let mut hi = vec![0_i32; count * pairs];
        let mut lo = vec![0_i32; count * pairs];
        for ((hi, lo), taps) in hi
            .chunks_exact_mut(pairs)
            .zip(lo.chunks_exact_mut(pairs))
            .zip(outputs)
        {
            for ((hi, lo), pair) in hi.iter_mut().zip(lo.iter_mut()).zip(taps.chunks(2)) {
                let (hi0, lo0) = split_tap(pair[0]);
                let (hi1, lo1) = pair.get(1).map_or((0, 0), |&tap| split_tap(tap));
                *hi = lane_pair(hi0, hi1);
                *lo = lane_pair(lo0, lo1);
            }
        }
        Self { hi, lo, pairs }
    }
}

/// Two 16-bit lanes as one `i32`, `low` in the low half.
#[cfg(target_arch = "x86_64")]
fn lane_pair(low: i16, high: i16) -> i32 {
    let ([a, b], [c, d]) = (low.to_le_bytes(), high.to_le_bytes());
    i32::from_le_bytes([a, b, c, d])
}

/// The width of a tap's byte chunks in the AVX-512 arms: a tap `t` is taken
/// as `c2 · 2^16 + c1 · 2^8 + c0` with every chunk a signed byte
/// ([`tap_chunks`]), the radix three signed bytes need for a 22-bit tap (a
/// weight of one is 2^22, past what chunks of seven bits reach), so that
/// `Σ pixel · t = ((Σ pixel · c2) << 8 + Σ pixel · c1) << 8 + Σ pixel · c0`
/// with every partial sum and every stage of that recombination inside `i32`
/// over the longest window of either filter
/// (`pil_tap_chunks_fit_the_bytes_and_32_bit_sums`).
#[cfg(any(target_arch = "x86_64", test))]
const CHUNK_BITS: u32 = 8;

/// `[c0, c1, c2]` of a tap, see [`CHUNK_BITS`]: each chunk is the low byte of
/// what the chunks below it leave, read as signed, so a chunk is in `[-128,
/// 127]` and the top one is below 2^7 in magnitude for a tap below 2^23
/// ([`FixedKernel`]).
#[cfg(any(target_arch = "x86_64", test))]
fn tap_chunks(tap: i32) -> [i8; 3] {
    let c0 = tap as i8;
    let rest = (tap - i32::from(c0)) >> CHUNK_BITS;
    let c1 = rest as i8;
    let c2 = (rest - i32::from(c1)) >> CHUNK_BITS;
    // Below 2^7 in magnitude by the kernel's bound; a tap outside it would be
    // a broken invariant, not a value to clamp.
    debug_assert!(
        i8::try_from(c2).is_ok(),
        "tap {tap} exceeds three byte chunks"
    );
    [c0, c1, c2 as i8]
}

/// The taps of one axis as the AVX-512 width pass reads them: the outputs in
/// groups of `outputs` (five for three channels, four for four, fewer when
/// the windows of a group would not fit one gather), the sixteen 32-bit
/// lanes of a vector being the group's outputs' channels. Per group, 64
/// bytes of gather indices (for every lane, the four consecutive source
/// pixels of its channel relative to the group's first window start, taken
/// four taps at a time from a 128-byte window that moves four pixels per
/// step), then per step of four taps three 64-byte blocks, one per chunk
/// level, in the lanes' layout (a lane's four bytes: its output's four
/// taps' chunks of that level), zero where a window ends or a lane has no
/// output.
#[cfg(target_arch = "x86_64")]
struct GroupTaps {
    outputs: usize,
    /// Bytes per group in `table`: the indices plus three blocks per step.
    stride: usize,
    table: Vec<u8>,
    /// Steps of four taps per group (the group's longest window).
    steps: Vec<usize>,
    /// Bytes of a row's end the pass copies behind zero padding, enough for
    /// the windows of the row's last groups: a gather of 128 bytes from the
    /// last step's start.
    tail: usize,
}

#[cfg(target_arch = "x86_64")]
impl GroupTaps {
    /// Bytes one gather reads from its window.
    const WINDOW: usize = 128;

    /// `None` when even pairs of outputs spread over more than one window
    /// (a downscale by more than about thirty), the arm's limit; the axis
    /// then takes the split-tap arm.
    fn new<const C: usize>(kernel: &FixedKernel) -> Option<Self> {
        let count = kernel.bounds.len();
        let longest = kernel.taps.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let most_steps = longest.div_ceil(4);
        let outputs = (2..=16 / C).rev().find(|&outputs| {
            kernel.bounds.chunks(outputs).all(|group| {
                let span = group[group.len() - 1].0 - group[0].0;
                (span + 4) * C <= Self::WINDOW
            })
        })?;
        let groups = count.div_ceil(outputs);
        let stride = 64 * (1 + 3 * most_steps);
        let mut table = vec![0_u8; groups * stride];
        let mut steps = Vec::with_capacity(groups);
        for ((g, group), block) in kernel
            .bounds
            .chunks(outputs)
            .enumerate()
            .zip(table.chunks_exact_mut(stride))
        {
            let first_start = group[0].0;
            let (indices, blocks) = block.split_at_mut(64);
            let mut group_longest = 1;
            for (o, &(start, n)) in group.iter().enumerate() {
                group_longest = group_longest.max(n);
                for c in 0..C {
                    let lane = o * C + c;
                    for (b, index) in indices[4 * lane..4 * lane + 4].iter_mut().enumerate() {
                        *index = ((start - first_start + b) * C + c) as u8;
                    }
                }
                for (k, &tap) in kernel.taps[g * outputs + o].iter().enumerate() {
                    for (level, chunk) in tap_chunks(tap).into_iter().enumerate() {
                        let block = &mut blocks[64 * (3 * (k / 4) + level)..][..64];
                        for c in 0..C {
                            block[4 * (o * C + c) + k % 4] = chunk as u8;
                        }
                    }
                }
            }
            steps.push(group_longest.div_ceil(4));
        }
        Some(Self {
            outputs,
            stride,
            table,
            steps,
            tail: Self::WINDOW + 4 * C * most_steps,
        })
    }
}

/// The taps of `count` output rows from `first` of one axis as the AVX-512
/// height pass reads them: per row and per step of four taps, three words
/// holding the four taps' chunks of one level each, a word the pass
/// broadcasts; zero past a window's end.
#[cfg(target_arch = "x86_64")]
struct ChunkTaps {
    words: Vec<i32>,
    /// Steps per row (the longest window of the rows).
    steps: usize,
}

#[cfg(target_arch = "x86_64")]
impl ChunkTaps {
    fn new(kernel: &FixedKernel, first: usize, count: usize) -> Self {
        let outputs = &kernel.taps[first..first + count];
        let steps = outputs
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .max(1)
            .div_ceil(4);
        let mut words = vec![0_i32; count * steps * 3];
        for (words, taps) in words.chunks_exact_mut(steps * 3).zip(outputs) {
            for (k, &tap) in taps.iter().enumerate() {
                for (level, chunk) in tap_chunks(tap).into_iter().enumerate() {
                    words[(k / 4) * 3 + level] |= i32::from(chunk as u8) << (8 * (k % 4));
                }
            }
        }
        Self { words, steps }
    }
}

/// Output rows `[y0, y0 + band.len() / row_len)` of the height pass, the
/// portable arm of [`HeightPass`]: each output row sums its taps over whole
/// source rows, one tap's row at a time, so the inner loop runs along the
/// row. `src` holds the source rows from `first_row` on.
fn pil_v_band(
    src: &[u8],
    first_row: usize,
    kernel: &FixedKernel,
    row_len: usize,
    y0: usize,
    band: &mut [u8],
) {
    let half = kernel.half();
    let mut sums = vec![0_i32; row_len];
    for (i, out_row) in band.chunks_exact_mut(row_len).enumerate() {
        let (start, n) = kernel.bounds[y0 + i];
        let start = start - first_row;
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

/// How [`pil_resample_vertical`] runs its rows: the portable row loop; on an
/// x86-64 processor that reports AVX2, the 16-bit row-pair steps of
/// [`avx2::v_band`] over the band's taps split for them; on one that
/// reports AVX-512 VNNI and VBMI, the byte dot products of
/// [`avx512::v_band`] over four source rows at a time.
#[derive(Clone, Copy)]
enum HeightPass {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl HeightPass {
    #[cfg(target_arch = "x86_64")]
    fn new() -> Self {
        if avx512::detected() {
            Self::Avx512
        } else if std::arch::is_x86_feature_detected!("avx2") {
            Self::Avx2
        } else {
            Self::Portable
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn new() -> Self {
        Self::Portable
    }

    /// Output rows `[y0, y0 + band.len() / row_len)` of the height pass over
    /// `src`, which holds the source rows from `first_row` on (every row the
    /// band's windows cover); a vector arm builds the band's tap layout,
    /// detects its features and calls in one place, as the crate's NEON
    /// kernels and the FMA pass keep them.
    #[cfg_attr(
        target_arch = "x86_64",
        expect(
            unsafe_code,
            reason = "the target-feature function runs only after this function detected its features"
        )
    )]
    fn rows(
        self,
        src: &[u8],
        first_row: usize,
        kernel: &FixedKernel,
        row_len: usize,
        y0: usize,
        band: &mut [u8],
    ) {
        match self {
            Self::Portable => pil_v_band(src, first_row, kernel, row_len, y0, band),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                if !std::arch::is_x86_feature_detected!("avx2") {
                    return pil_v_band(src, first_row, kernel, row_len, y0, band);
                }
                let split = SplitTaps::new(kernel, y0, band.len() / row_len, 2);
                // SAFETY: `is_x86_feature_detected!("avx2")` answered true just
                // above, the only requirement of a `target_feature(enable =
                // "avx2")` function.
                unsafe { avx2::v_band(src, first_row, kernel, &split, row_len, y0, band) }
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                if !avx512::detected() {
                    return pil_v_band(src, first_row, kernel, row_len, y0, band);
                }
                let chunks = ChunkTaps::new(kernel, y0, band.len() / row_len);
                // SAFETY: `avx512::detected()` answered true just above: the
                // processor reports every feature the `target_feature`
                // function enables, its only requirement.
                unsafe { avx512::v_band(src, first_row, kernel, &chunks, row_len, y0, band) }
            }
        }
    }
}

/// The Pillow-exact passes compiled for AVX2, the vector arms of
/// [`WidthPass`] and [`HeightPass`]: the same integer sums as the portable
/// passes, taken two taps at a time by the multiply-add of adjacent 16-bit
/// lanes over the taps' split halves ([`SplitTaps`]) and recombined in 32
/// bits per output. Integer sums are the same in any order, so the bytes
/// are those of the portable passes and of Pillow.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    #![expect(
        unsafe_code,
        reason = "the vector loads and stores; every pointer is bounded by an exact-size chunk of a slice"
    )]

    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
        _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_packs_epi32, _mm256_packus_epi16,
        _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_set_m128i, _mm256_setr_epi8,
        _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi32, _mm256_sra_epi32,
        _mm256_storeu_si256, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16, _mm_cvtsi128_si32,
        _mm_cvtsi32_si128, _mm_loadu_si128, _mm_setr_epi8, _mm_shuffle_epi8, _mm_storeu_si128,
    };

    use super::{BandScratch, FixedKernel, SplitTaps, SPLIT_LOW_BITS};

    /// Pixels a scratch row carries past the source row: a window padded to
    /// a multiple of four taps reads at most three pixels past its last tap.
    const PADDING_PIXELS: usize = 4;

    /// Output rows `[y0, y0 + band.len() / (out_w * C))` of the width pass,
    /// two source rows per step. Each row is spread to four bytes per pixel
    /// in a zero-padded scratch row; per output, every four source pixels of
    /// the two rows load as one vector (a row per 128-bit half), two
    /// shuffles pair the bytes of pixels (0, 1) and (2, 3) channel by
    /// channel as 16-bit lanes, and the multiply-adds against the taps'
    /// pairs `(t0, t1)` and `(t2, t3)` sum two pixels of one channel per
    /// lane, the high and the low halves of the taps into accumulators of
    /// their own. A band's odd last row runs alone in both halves.
    #[target_feature(enable = "avx2")]
    #[expect(
        clippy::too_many_arguments,
        reason = "the pass's scratch rides along so every band of a pass reuses it"
    )]
    pub(super) fn h_band<const C: usize>(
        src: &[u8],
        kernel: &FixedKernel,
        split: &SplitTaps,
        in_w: usize,
        out_w: usize,
        y0: usize,
        band: &mut [u8],
        scratch: &mut BandScratch,
    ) {
        let row_in = in_w * C;
        let row_out = out_w * C;
        let scratch_len = (in_w + PADDING_PIXELS) * 4;
        let scratch = scratch.bytes(2 * scratch_len);
        let (first, second) = scratch.split_at_mut(scratch_len);
        // The spread rows are read PADDING_PIXELS past their end as zeros;
        // zero that padding here, so the kernel owes nothing to what the
        // scratch held before this band.
        first[in_w * 4..].fill(0);
        second[in_w * 4..].fill(0);
        let mut pairs = band.chunks_exact_mut(2 * row_out);
        let mut y = y0;
        for pair in pairs.by_ref() {
            spread::<C>(&src[y * row_in..(y + 1) * row_in], first);
            spread::<C>(&src[(y + 1) * row_in..(y + 2) * row_in], second);
            let (out_a, out_b) = pair.split_at_mut(row_out);
            h_outputs::<C>(first, second, kernel, split, out_a, Some(out_b));
            y += 2;
        }
        let rest = pairs.into_remainder();
        if !rest.is_empty() {
            spread::<C>(&src[y * row_in..(y + 1) * row_in], first);
            h_outputs::<C>(first, first, kernel, split, rest, None);
        }
    }

    /// `row` (`C`-channel pixels) as four bytes per pixel at the start of
    /// `rgbx`, the fourth byte zero for three channels; the bytes past the
    /// row keep the scratch's zeros.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn spread<const C: usize>(row: &[u8], rgbx: &mut [u8]) {
        if C == 4 {
            rgbx[..row.len()].copy_from_slice(row);
            return;
        }
        // Four pixels (12 bytes) spread to 16 per step; the step reads 16
        // bytes, so every group whose read ends inside the row is done here
        // (`12 * groups + 4 <= row.len()`, which is what `first_chunk` finds)
        // and the last pixels one by one. The scratch is resliced to the
        // stores' exact length first, so a short one panics on the bound
        // instead of being written past.
        let spread = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
        let groups = row.len().saturating_sub(4) / 12;
        let (wide, rest) = rgbx.split_at_mut(16 * groups);
        for (group, wide) in wide.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            if let Some(bytes) = row[12 * group..].first_chunk::<16>() {
                store16(wide, _mm_shuffle_epi8(load16(bytes), spread));
            }
        }
        for (pixel, wide) in row[12 * groups..]
            .as_chunks::<3>()
            .0
            .iter()
            .zip(rest.as_chunks_mut::<4>().0.iter_mut())
        {
            wide[..3].copy_from_slice(pixel);
            wide[3] = 0;
        }
    }

    /// Sixteen bytes stored.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn store16(out: &mut [u8; 16], bytes: __m128i) {
        // SAFETY: the array is exactly 16 bytes.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast::<__m128i>(), bytes) }
    }

    /// Every output of the two spread rows `row_a` (the low halves) and
    /// `row_b` (the high halves) into `out_a` and, when given, `out_b`.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn h_outputs<const C: usize>(
        row_a: &[u8],
        row_b: &[u8],
        kernel: &FixedKernel,
        split: &SplitTaps,
        out_a: &mut [u8],
        mut out_b: Option<&mut [u8]>,
    ) {
        // Per 128-bit half, the bytes of pixels 0 and 1 (`first`), then 2
        // and 3 (`second`), of a 16-byte group as 16-bit lanes paired
        // channel by channel; `-1` zero-fills the high byte of a lane.
        let first = _mm256_setr_epi8(
            0, -1, 4, -1, 1, -1, 5, -1, 2, -1, 6, -1, 3, -1, 7, -1, 0, -1, 4, -1, 1, -1, 5, -1, 2,
            -1, 6, -1, 3, -1, 7, -1,
        );
        let second = _mm256_setr_epi8(
            8, -1, 12, -1, 9, -1, 13, -1, 10, -1, 14, -1, 11, -1, 15, -1, 8, -1, 12, -1, 9, -1, 13,
            -1, 10, -1, 14, -1, 11, -1, 15, -1,
        );
        let half = _mm256_set1_epi32(kernel.half());
        let shift = _mm_cvtsi32_si128(kernel.precision as i32);
        let zero = _mm256_setzero_si256();
        for (x, &(x0, n)) in kernel.bounds.iter().enumerate() {
            let groups = n.div_ceil(4);
            let hi = &split.hi[x * split.pairs..x * split.pairs + 2 * groups];
            let lo = &split.lo[x * split.pairs..x * split.pairs + 2 * groups];
            let source_a = &row_a[x0 * 4..(x0 + 4 * groups) * 4];
            let source_b = &row_b[x0 * 4..(x0 + 4 * groups) * 4];
            let mut sum_hi = zero;
            let mut sum_lo = half;
            for (((a, b), hi), lo) in source_a
                .as_chunks::<16>()
                .0
                .iter()
                .zip(source_b.as_chunks::<16>().0)
                .zip(hi.as_chunks::<2>().0)
                .zip(lo.as_chunks::<2>().0)
            {
                let pixels = _mm256_set_m128i(load16(b), load16(a));
                let pixels01 = _mm256_shuffle_epi8(pixels, first);
                let pixels23 = _mm256_shuffle_epi8(pixels, second);
                sum_hi = _mm256_add_epi32(
                    sum_hi,
                    _mm256_madd_epi16(pixels01, _mm256_set1_epi32(hi[0])),
                );
                sum_lo = _mm256_add_epi32(
                    sum_lo,
                    _mm256_madd_epi16(pixels01, _mm256_set1_epi32(lo[0])),
                );
                sum_hi = _mm256_add_epi32(
                    sum_hi,
                    _mm256_madd_epi16(pixels23, _mm256_set1_epi32(hi[1])),
                );
                sum_lo = _mm256_add_epi32(
                    sum_lo,
                    _mm256_madd_epi16(pixels23, _mm256_set1_epi32(lo[1])),
                );
            }
            let shifted = recombined(sum_hi, sum_lo, shift);
            // Both packs saturate: `i32 -> i16` keeps the shifted sums (below
            // 2^10 in magnitude), `i16 -> u8` is the clamp to a byte; each
            // half's pixel sits in its low four bytes.
            let bytes = _mm256_packus_epi16(_mm256_packs_epi32(shifted, shifted), zero);
            store::<C>(
                out_a,
                x,
                _mm_cvtsi128_si32(_mm256_castsi256_si128(bytes)).to_le_bytes(),
            );
            if let Some(out_b) = out_b.as_deref_mut() {
                store::<C>(
                    out_b,
                    x,
                    _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(bytes)).to_le_bytes(),
                );
            }
        }
    }

    /// Output `x`'s `C` bytes of `word` into its row: a four-byte write when
    /// the next pixel's first byte follows (the next output writes it
    /// again), else the last pixel's three.
    #[inline]
    fn store<const C: usize>(out_row: &mut [u8], x: usize, word: [u8; 4]) {
        let at = x * C;
        if C == 4 || at + 4 <= out_row.len() {
            out_row[at..at + 4].copy_from_slice(&word);
        } else {
            out_row[at..at + 3].copy_from_slice(&word[..3]);
        }
    }

    /// `(sum_hi << SPLIT_LOW_BITS) + sum_lo`, the sums of a tap's halves
    /// recombined, shifted by the kernel's precision: `kernel.clip8` before
    /// its clamp, eight lanes at a time.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn recombined(sum_hi: __m256i, sum_lo: __m256i, shift: __m128i) -> __m256i {
        _mm256_sra_epi32(
            _mm256_add_epi32(
                _mm256_slli_epi32::<{ SPLIT_LOW_BITS as i32 }>(sum_hi),
                sum_lo,
            ),
            shift,
        )
    }

    /// Sixteen bytes as one 128-bit vector.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn load16(bytes: &[u8; 16]) -> __m128i {
        // SAFETY: the array is exactly 16 bytes.
        unsafe { _mm_loadu_si128(bytes.as_ptr().cast::<__m128i>()) }
    }

    /// Output rows `[y0, y0 + band.len() / row_len)` of the height pass,
    /// thirty-two columns at a time across the taps, two source rows per
    /// step: each row's bytes widen to 16-bit lanes, the two rows interleave
    /// word by word so that adjacent lanes hold one column's two pixels, and
    /// the multiply-adds against the taps' pair `(t0, t1)` sum them, the high
    /// and the low halves of the taps into accumulators of their own; an odd
    /// last tap pairs with a zero tap over its own row. The row's last
    /// columns run in scalar. `src` holds the source rows from `first_row`
    /// on.
    #[target_feature(enable = "avx2")]
    pub(super) fn v_band(
        src: &[u8],
        first_row: usize,
        kernel: &FixedKernel,
        split: &SplitTaps,
        row_len: usize,
        y0: usize,
        band: &mut [u8],
    ) {
        let half = _mm256_set1_epi32(kernel.half());
        let shift = _mm_cvtsi32_si128(kernel.precision as i32);
        let zero = _mm256_setzero_si256();
        for (i, out_row) in band.chunks_exact_mut(row_len).enumerate() {
            let (start, n) = kernel.bounds[y0 + i];
            let start = start - first_row;
            let pairs = n.div_ceil(2);
            let hi = &split.hi[i * split.pairs..i * split.pairs + pairs];
            let lo = &split.lo[i * split.pairs..i * split.pairs + pairs];
            let (chunks, rest) = out_row.as_chunks_mut::<32>();
            for (x, out) in chunks.iter_mut().enumerate() {
                let mut sums_hi = [zero; 4];
                let mut sums_lo = [half; 4];
                for (k, (&hi, &lo)) in hi.iter().zip(lo).enumerate() {
                    let row_a = start + 2 * k;
                    let row_b = (row_a + 1).min(start + n - 1);
                    let a = src[row_a * row_len + x * 32..][..32].as_chunks::<16>().0;
                    let b = src[row_b * row_len + x * 32..][..32].as_chunks::<16>().0;
                    let (a0, a1) = (widened(&a[0]), widened(&a[1]));
                    let (b0, b1) = (widened(&b[0]), widened(&b[1]));
                    // Columns 0..4 and 8..12 of the first sixteen, then 4..8
                    // and 12..16, then the same of the second sixteen: the
                    // order the packs below put back.
                    let columns = [
                        _mm256_unpacklo_epi16(a0, b0),
                        _mm256_unpackhi_epi16(a0, b0),
                        _mm256_unpacklo_epi16(a1, b1),
                        _mm256_unpackhi_epi16(a1, b1),
                    ];
                    let (hi, lo) = (_mm256_set1_epi32(hi), _mm256_set1_epi32(lo));
                    for ((sum_hi, sum_lo), columns) in
                        sums_hi.iter_mut().zip(&mut sums_lo).zip(columns)
                    {
                        *sum_hi = _mm256_add_epi32(*sum_hi, _mm256_madd_epi16(columns, hi));
                        *sum_lo = _mm256_add_epi32(*sum_lo, _mm256_madd_epi16(columns, lo));
                    }
                }
                let [s0, s1, s2, s3] =
                    std::array::from_fn(|q| recombined(sums_hi[q], sums_lo[q], shift));
                // Both packs saturate (`i32 -> i16` keeps the shifted sums,
                // `i16 -> u8` is the clamp); packing (0..4, 8..12) with (4..8,
                // 12..16) puts a half's sixteen columns in order, the byte
                // pack leaves the four eight-byte quarters as 0, 2, 1, 3.
                let bytes =
                    _mm256_packus_epi16(_mm256_packs_epi32(s0, s1), _mm256_packs_epi32(s2, s3));
                store32(out, _mm256_permute4x64_epi64::<0b11_01_10_00>(bytes));
            }
            let done = chunks.len() * 32;
            for (j, out) in rest.iter_mut().enumerate() {
                let column = done + j;
                let mut sum = kernel.half();
                for (k, &tap) in kernel.taps[y0 + i].iter().take(n).enumerate() {
                    sum += i32::from(src[(start + k) * row_len + column]) * tap;
                }
                *out = kernel.clip8(sum);
            }
        }
    }

    /// Sixteen bytes widened to 16-bit lanes.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn widened(bytes: &[u8; 16]) -> __m256i {
        _mm256_cvtepu8_epi16(load16(bytes))
    }

    /// Thirty-two bytes stored.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn store32(out: &mut [u8; 32], bytes: __m256i) {
        // SAFETY: the array is exactly 32 bytes.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast::<__m256i>(), bytes) }
    }
}

/// The Pillow-exact passes compiled for the AVX-512 VNNI unit with VBMI, the
/// vector arms of [`WidthPass`] and [`HeightPass`] where the processor has
/// one: the same integer sums as the portable passes, taken four taps at a
/// time by the byte dot product (sixteen lanes of four unsigned pixel bytes
/// against four signed tap bytes, accumulated in 32 bits without saturation)
/// over the taps' three byte chunks ([`tap_chunks`]), an accumulator per
/// chunk level, recombined per output in 32 bits. Integer sums are the same
/// in any order, so the bytes are those of the portable passes and of
/// Pillow.
#[cfg(target_arch = "x86_64")]
mod avx512 {
    #![expect(
        unsafe_code,
        reason = "the vector loads and stores; every pointer is bounded by an exact-size chunk of a slice or by a mask over the bytes of a slice"
    )]

    use std::arch::x86_64::{
        __m128i, __m512i, __mmask64, _mm512_add_epi32, _mm512_dpbusd_epi32, _mm512_loadu_epi32,
        _mm512_loadu_epi8, _mm512_mask_storeu_epi8, _mm512_maskz_loadu_epi8, _mm512_packus_epi16,
        _mm512_packus_epi32, _mm512_permutex2var_epi8, _mm512_permutexvar_epi32, _mm512_set1_epi32,
        _mm512_setr_epi32, _mm512_setzero_si512, _mm512_slli_epi32, _mm512_sra_epi32,
        _mm512_storeu_epi32, _mm512_storeu_epi8, _mm512_unpackhi_epi16, _mm512_unpackhi_epi8,
        _mm512_unpacklo_epi16, _mm512_unpacklo_epi8, _mm_cvtsi32_si128,
    };

    use super::{BandScratch, ChunkTaps, FixedKernel, GroupTaps, CHUNK_BITS};

    /// The processor reports every feature the arms enable.
    pub(super) fn detected() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
    }

    /// Output rows a group iteration of the width pass carries at once: the
    /// group's gather indices and every step's chunk lines are loaded once
    /// for them, and their dot products are independent chains.
    const ROWS: usize = 4;

    /// Output rows `[y0, y0 + band.len() / (out_w * C))` of the width pass,
    /// [`ROWS`] rows at a time by [`h_rows`] and the last rows one by one.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[expect(
        clippy::too_many_arguments,
        reason = "the pass's scratch rides along so every band of a pass reuses it"
    )]
    pub(super) fn h_band<const C: usize>(
        src: &[u8],
        kernel: &FixedKernel,
        groups: &GroupTaps,
        in_w: usize,
        out_w: usize,
        y0: usize,
        band: &mut [u8],
        scratch: &mut BandScratch,
    ) {
        let row_in = in_w * C;
        let row_out = out_w * C;
        let tail_len = groups.tail.min(row_in);
        // A group's sixteen lanes land whole; the last group's lanes past the
        // row and the narrowing's last loads stay inside a row's scratch.
        let (tails, sums) = scratch.parts(
            ROWS * (tail_len + GroupTaps::WINDOW),
            ROWS * (row_out.next_multiple_of(64) + 16),
        );
        // Each tail row is read WINDOW bytes past its end as zeros; zero that
        // padding here, so the kernel owes nothing to what the scratch held
        // before this band (the sums need no zeros: the narrowing's masked
        // store drops the lanes past the row).
        for tail in tails.chunks_exact_mut(tail_len + GroupTaps::WINDOW) {
            tail[tail_len..].fill(0);
        }
        let mut blocks = band.chunks_exact_mut(ROWS * row_out);
        let mut y = y0;
        for block in blocks.by_ref() {
            h_rows::<C, ROWS>(src, kernel, groups, row_in, y, block, tails, sums);
            y += ROWS;
        }
        for out_row in blocks.into_remainder().chunks_exact_mut(row_out) {
            h_rows::<C, 1>(src, kernel, groups, row_in, y, out_row, tails, sums);
            y += 1;
        }
    }

    /// Output rows `[y0, y0 + R)` of the width pass into `out`, a group of
    /// outputs per vector ([`GroupTaps`]): per step of four taps, each row's
    /// two vectors from the group's window start are gathered by the group's
    /// indices into every lane's four pixels and three dot products
    /// accumulate them against the step's chunk lines; the windows of a
    /// row's last groups read from a zero-padded copy of the row's end (one
    /// per row in `tails`). A group's recombined and shifted sums go to the
    /// row's scratch of 32-bit lanes in `sums`, which [`narrowed`] clamps
    /// into the output row once the row's groups are done.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    #[expect(
        clippy::too_many_arguments,
        reason = "the band's scratch buffers ride along so every block of rows reuses them"
    )]
    fn h_rows<const C: usize, const R: usize>(
        src: &[u8],
        kernel: &FixedKernel,
        groups: &GroupTaps,
        row_in: usize,
        y0: usize,
        out: &mut [u8],
        tails: &mut [u8],
        sums: &mut [i32],
    ) {
        let row_out = out.len() / R;
        let tail_stride = tails.len() / ROWS;
        let tail_len = tail_stride - GroupTaps::WINDOW;
        let scratch = sums.len() / ROWS;
        let half = _mm512_set1_epi32(kernel.half());
        let shift = _mm_cvtsi32_si128(kernel.precision as i32);
        let zero = _mm512_setzero_si512();
        let mut rows: [&[u8]; R] = [&[]; R];
        for (r, row) in rows.iter_mut().enumerate() {
            *row = &src[(y0 + r) * row_in..(y0 + r + 1) * row_in];
        }
        for (tail, row) in tails.chunks_exact_mut(tail_stride).zip(&rows) {
            tail[..tail_len].copy_from_slice(&row[row_in - tail_len..]);
        }
        let tails: &[u8] = tails;
        let mut padded: [&[u8]; R] = [&[]; R];
        for (r, padded) in padded.iter_mut().enumerate() {
            *padded = &tails[r * tail_stride..(r + 1) * tail_stride];
        }
        for ((g, block), &steps) in groups
            .table
            .chunks_exact(groups.stride)
            .enumerate()
            .zip(&groups.steps)
        {
            let first = g * groups.outputs;
            let begin = kernel.bounds[first].0 * C;
            // The rows themselves while the group's last window ends inside
            // them, else the padded copies of their ends.
            let (sources, base) = if begin + 4 * C * (steps - 1) + GroupTaps::WINDOW <= row_in {
                (&rows, begin)
            } else {
                (&padded, begin - (row_in - tail_len))
            };
            let (indices, blocks) = block.split_at(64);
            let indices = load64(&indices.as_chunks::<64>().0[0]);
            let (mut sum0, mut sum1, mut sum2) = ([half; R], [zero; R], [zero; R]);
            for (s, levels) in blocks.as_chunks::<192>().0.iter().take(steps).enumerate() {
                let at = base + 4 * C * s;
                let levels = levels.as_chunks::<64>().0;
                let (chunks0, chunks1, chunks2) =
                    (load64(&levels[0]), load64(&levels[1]), load64(&levels[2]));
                for (((source, sum0), sum1), sum2) in
                    sources.iter().zip(&mut sum0).zip(&mut sum1).zip(&mut sum2)
                {
                    let window = source[at..at + GroupTaps::WINDOW].as_chunks::<64>().0;
                    let pixels =
                        _mm512_permutex2var_epi8(load64(&window[0]), indices, load64(&window[1]));
                    *sum0 = _mm512_dpbusd_epi32(*sum0, pixels, chunks0);
                    *sum1 = _mm512_dpbusd_epi32(*sum1, pixels, chunks1);
                    *sum2 = _mm512_dpbusd_epi32(*sum2, pixels, chunks2);
                }
            }
            let at = first * C;
            for (r, ((&sum0, &sum1), &sum2)) in sum0.iter().zip(&sum1).zip(&sum2).enumerate() {
                store16(
                    &mut sums[r * scratch + at..r * scratch + at + 16]
                        .as_chunks_mut::<16>()
                        .0[0],
                    recombined(sum0, sum1, sum2, shift),
                );
            }
        }
        for (sums, out_row) in sums
            .chunks_exact(scratch)
            .zip(out.chunks_exact_mut(row_out))
        {
            narrowed(sums, out_row);
        }
    }

    /// `out`'s row of shifted sums (`sums`, at least as long rounded up to a
    /// multiple of sixty-four) clamped to bytes, sixty-four at a time: both
    /// packs saturate (`i32 -> u16` is the clamp below zero, `i16 -> u8` the
    /// clamp above 255; a shifted sum is below 2^10 in magnitude) and leave
    /// every 128-bit lane with four bytes of each of the four source
    /// vectors, which one dword permute puts in order; the row's last
    /// columns run under a mask.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    fn narrowed(sums: &[i32], out: &mut [u8]) {
        let order = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        for (sums, out) in sums.as_chunks::<64>().0.iter().zip(out.chunks_mut(64)) {
            let quads = sums.as_chunks::<16>().0;
            let bytes = _mm512_packus_epi16(
                _mm512_packus_epi32(load16(&quads[0]), load16(&quads[1])),
                _mm512_packus_epi32(load16(&quads[2]), load16(&quads[3])),
            );
            let bytes = _mm512_permutexvar_epi32(order, bytes);
            match out.as_chunks_mut::<64>() {
                ([full], []) => store64(full, bytes),
                _ => store_masked(out, bytes),
            }
        }
    }

    /// Output rows `[y0, y0 + band.len() / row_len)` of the height pass,
    /// sixty-four columns at a time across the taps ([`v_tile`]); the row's
    /// last columns run under a mask. `src` holds the source rows from
    /// `first_row` on.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    pub(super) fn v_band(
        src: &[u8],
        first_row: usize,
        kernel: &FixedKernel,
        chunks: &ChunkTaps,
        row_len: usize,
        y0: usize,
        band: &mut [u8],
    ) {
        for (i, out_row) in band.chunks_exact_mut(row_len).enumerate() {
            let (start, n) = kernel.bounds[y0 + i];
            let start = start - first_row;
            let window = (start, start + n - 1);
            let words = &chunks.words[i * chunks.steps * 3..][..n.div_ceil(4) * 3];
            let (full, rest) = out_row.as_chunks_mut::<64>();
            for (x, out) in full.iter_mut().enumerate() {
                v_tile::<false>(src, kernel, row_len, window, words, x * 64, out);
            }
            if !rest.is_empty() {
                v_tile::<true>(src, kernel, row_len, window, words, full.len() * 64, rest);
            }
        }
    }

    /// One output row's columns `column..column + out.len()` (sixty-four, or
    /// fewer under the mask) of the height pass, four source rows per step:
    /// the rows' bytes interleave (bytes, then words) so that every 32-bit
    /// lane holds one column's four pixels in row order, and the dot
    /// products against the step's broadcast chunks accumulate, a level in
    /// accumulators of its own; the rows past the window's last (`rows.1`)
    /// repeat it under zero chunks.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    fn v_tile<const MASKED: bool>(
        src: &[u8],
        kernel: &FixedKernel,
        row_len: usize,
        rows: (usize, usize),
        words: &[i32],
        column: usize,
        out: &mut [u8],
    ) {
        let half = _mm512_set1_epi32(kernel.half());
        let shift = _mm_cvtsi32_si128(kernel.precision as i32);
        let zero = _mm512_setzero_si512();
        let (start, last) = rows;
        let mut sums = [[half; 4], [zero; 4], [zero; 4]];
        for (s, levels) in words.as_chunks::<3>().0.iter().enumerate() {
            let mut rows = [zero; 4];
            for (j, row) in rows.iter_mut().enumerate() {
                let y = (start + 4 * s + j).min(last);
                let bytes = &src[y * row_len + column..][..out.len()];
                *row = if MASKED {
                    load_masked(bytes)
                } else {
                    load64(&bytes.as_chunks::<64>().0[0])
                };
            }
            let (ab_lo, ab_hi) = (
                _mm512_unpacklo_epi8(rows[0], rows[1]),
                _mm512_unpackhi_epi8(rows[0], rows[1]),
            );
            let (cd_lo, cd_hi) = (
                _mm512_unpacklo_epi8(rows[2], rows[3]),
                _mm512_unpackhi_epi8(rows[2], rows[3]),
            );
            // Columns 0..4, 4..8, 8..12 and 12..16 of each sixteen: the
            // order the packs below put back.
            let tiles = [
                _mm512_unpacklo_epi16(ab_lo, cd_lo),
                _mm512_unpackhi_epi16(ab_lo, cd_lo),
                _mm512_unpacklo_epi16(ab_hi, cd_hi),
                _mm512_unpackhi_epi16(ab_hi, cd_hi),
            ];
            for (sums, &word) in sums.iter_mut().zip(levels) {
                let taps = _mm512_set1_epi32(word);
                for (sum, &tile) in sums.iter_mut().zip(&tiles) {
                    *sum = _mm512_dpbusd_epi32(*sum, tile, taps);
                }
            }
        }
        let [sum0, sum1, sum2] = sums;
        let (s0, s1, s2, s3) = (
            recombined(sum0[0], sum1[0], sum2[0], shift),
            recombined(sum0[1], sum1[1], sum2[1], shift),
            recombined(sum0[2], sum1[2], sum2[2], shift),
            recombined(sum0[3], sum1[3], sum2[3], shift),
        );
        // Both packs saturate (the clamp); packing the four-column tiles in
        // pairs, then the pairs, puts every 128-bit lane's sixteen columns
        // in order.
        let bytes = _mm512_packus_epi16(_mm512_packus_epi32(s0, s1), _mm512_packus_epi32(s2, s3));
        if MASKED {
            store_masked(out, bytes);
        } else {
            store64(&mut out.as_chunks_mut::<64>().0[0], bytes);
        }
    }

    /// `((sum2 << 8) + sum1) << 8 + sum0`, the sums of a tap's chunks
    /// recombined, shifted by the kernel's precision: `kernel.clip8` before
    /// its clamp, sixteen lanes at a time.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    fn recombined(sum0: __m512i, sum1: __m512i, sum2: __m512i, shift: __m128i) -> __m512i {
        let high = _mm512_add_epi32(_mm512_slli_epi32::<CHUNK_BITS>(sum2), sum1);
        _mm512_sra_epi32(
            _mm512_add_epi32(_mm512_slli_epi32::<CHUNK_BITS>(high), sum0),
            shift,
        )
    }

    /// Sixty-four bytes as one vector.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    fn load64(bytes: &[u8; 64]) -> __m512i {
        // SAFETY: the array is exactly 64 bytes.
        unsafe { _mm512_loadu_epi8(bytes.as_ptr().cast::<i8>()) }
    }

    /// Sixty-four bytes stored.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    fn store64(out: &mut [u8; 64], bytes: __m512i) {
        // SAFETY: the array is exactly 64 bytes.
        unsafe { _mm512_storeu_epi8(out.as_mut_ptr().cast::<i8>(), bytes) }
    }

    /// The bytes a mask over `len` bytes of a vector selects.
    #[inline]
    fn mask64(len: usize) -> __mmask64 {
        debug_assert!(len <= 64, "{len} bytes exceed a vector");
        if len >= 64 {
            !0
        } else {
            (1_u64 << len) - 1
        }
    }

    /// At most sixty-four bytes as one vector, zero past them.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    fn load_masked(bytes: &[u8]) -> __m512i {
        // SAFETY: the mask selects exactly the slice's bytes and a masked
        // load reads no byte outside its mask.
        unsafe { _mm512_maskz_loadu_epi8(mask64(bytes.len()), bytes.as_ptr().cast::<i8>()) }
    }

    /// The first `out.len()` (at most sixty-four) bytes of a vector stored.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    fn store_masked(out: &mut [u8], bytes: __m512i) {
        // SAFETY: the mask selects exactly the slice's bytes and a masked
        // store writes no byte outside its mask.
        unsafe { _mm512_mask_storeu_epi8(out.as_mut_ptr().cast::<i8>(), mask64(out.len()), bytes) }
    }

    /// Sixteen 32-bit lanes as one vector.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    fn load16(lanes: &[i32; 16]) -> __m512i {
        // SAFETY: the array is exactly sixteen 32-bit lanes.
        unsafe { _mm512_loadu_epi32(lanes.as_ptr()) }
    }

    /// Sixteen 32-bit lanes stored.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")]
    #[inline]
    fn store16(out: &mut [i32; 16], lanes: __m512i) {
        // SAFETY: the array is exactly sixteen 32-bit lanes.
        unsafe { _mm512_storeu_epi32(out.as_mut_ptr(), lanes) }
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
    let mut out = plain_buffer(rows * row_out);
    let pass = &WidthPass::new::<C>(kernel);
    let tasks = par_threads(out.len(), rows);
    if tasks <= 1 {
        pass.rows::<C>(
            src,
            kernel,
            in_w,
            out_w,
            0,
            &mut out,
            &mut BandScratch::default(),
        );
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
            scope.spawn(move |_| {
                pass.rows::<C>(
                    src,
                    kernel,
                    in_w,
                    out_w,
                    start,
                    band,
                    &mut BandScratch::default(),
                );
            });
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
    let mut out = plain_buffer(out_h * row_len);
    let pass = HeightPass::new();
    let tasks = par_threads(out.len(), out_h);
    if tasks <= 1 {
        pass.rows(src, 0, kernel, row_len, 0, &mut out);
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
            scope.spawn(move |_| pass.rows(src, 0, kernel, row_len, start, band));
            y0 += n;
        }
    });
    out
}

/// Bytes of width-pass rows a band of the height pass reads at a time when
/// both axes change ([`pil_resample_both`]): a band's rows stay in a core's
/// second-level cache between the pass that writes them and the pass that
/// reads them, in place of an intermediate image of every source row (88 MB
/// for an 8192x8192 image resized to 3584x3584), which was paged in fresh per
/// image, written once and read back from memory.
const PIL_BAND_BYTES: usize = 512 * 1024;

/// Source rows the width pass's vector arms take per block; a band's new
/// rows end on a block so that no band ends in an arm's one-row tail.
const PIL_WIDTH_ROWS: usize = 4;

/// The two passes of a resample that changes both axes, run in bands of
/// output rows: a band's height pass reads the width-pass rows its windows
/// cover, which the width pass wrote just before into a buffer of about
/// [`PIL_BAND_BYTES`] (one output row's windows at least); the rows the next
/// band shares with it are carried over, not resampled again. Every output
/// byte is the same two sums as over a whole intermediate image, so the bytes
/// are those of the per-sample reference.
struct BandedPasses<'a, const C: usize> {
    src: &'a [u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    width: &'a FixedKernel,
    height: &'a FixedKernel,
    width_pass: WidthPass,
    height_pass: HeightPass,
    band_bytes: usize,
}

impl<const C: usize> BandedPasses<'_, C> {
    /// Output rows `[y0, y0 + out.len() / (out_w * C))`, a band at a time.
    fn rows(&self, y0: usize, mut out: &mut [u8]) {
        let row_len = self.out_w * C;
        // The width-pass rows `mid` holds: source rows `[mid_start, mid_end)`.
        let mut mid = Vec::new();
        let mut scratch = BandScratch::default();
        let (mut mid_start, mut mid_end) = (0, 0);
        let mut y = y0;
        let end = y0 + out.len() / row_len;
        while y < end {
            // The band `[y, y1)`: as many output rows as keep their windows'
            // source rows within the budget, the first row whatever its
            // window takes.
            let (start, n) = self.height.bounds[y];
            let mut stop = start + n;
            let mut y1 = y + 1;
            while y1 < end {
                let (next_start, next_n) = self.height.bounds[y1];
                let next_stop = stop.max(next_start + next_n);
                if (next_stop - start) * row_len > self.band_bytes {
                    break;
                }
                stop = next_stop;
                y1 += 1;
            }
            // The band's rows end on a block; those already in `mid` move to
            // its front (windows start in order, so the band starts at or
            // after the buffer's rows) and the width pass fills in the rest.
            let stop = stop.next_multiple_of(PIL_WIDTH_ROWS).min(self.in_h);
            let kept = mid_end.clamp(start, stop);
            if kept > start {
                mid.copy_within(
                    (start - mid_start) * row_len..(kept - mid_start) * row_len,
                    0,
                );
            }
            mid.resize((stop - start) * row_len, 0);
            if kept < stop {
                self.width_pass.rows::<C>(
                    self.src,
                    self.width,
                    self.in_w,
                    self.out_w,
                    kept,
                    &mut mid[(kept - start) * row_len..],
                    &mut scratch,
                );
            }
            (mid_start, mid_end) = (start, stop);
            let (band, rest) = out.split_at_mut((y1 - y) * row_len);
            self.height_pass
                .rows(&mid, mid_start, self.height, row_len, y, band);
            out = rest;
            y = y1;
        }
    }
}

/// `src` (`in_w` x `in_h` interleaved pixels of `C` channels) resampled on
/// both axes to `out_w` x `out_h` by [`BandedPasses`] in bands of about
/// `band_bytes`; the output rows are banded across the pool, each task with a
/// buffer of its own.
fn pil_resample_both<const C: usize>(
    src: &[u8],
    (in_w, in_h): (usize, usize),
    (out_w, out_h): (usize, usize),
    width: &FixedKernel,
    height: &FixedKernel,
    band_bytes: usize,
) -> Vec<u8> {
    let row_len = out_w * C;
    let mut out = plain_buffer(out_h * row_len);
    let passes = &BandedPasses::<C> {
        src,
        in_w,
        in_h,
        out_w,
        width,
        height,
        width_pass: WidthPass::new::<C>(width),
        height_pass: HeightPass::new(),
        band_bytes,
    };
    let tasks = par_threads(out.len(), out_h);
    if tasks <= 1 {
        passes.rows(0, &mut out);
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
            scope.spawn(move |_| passes.rows(start, band));
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
    if width == 0 || height == 0 {
        return Err(TransformError::ShapeError(format!(
            "PIL bicubic RGB source {width}x{height} must be non-empty"
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

/// Pillow-exact BILINEAR resize over borrowed interleaved RGB bytes: the
/// bytes of [`resize_bilinear_pil`], for a processor that resizes a decoded
/// image's bytes more than once before it converts them.
pub fn resize_bilinear_pil_rgb(
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
            "PIL bilinear RGB source has {} bytes, expected {expected} for {width}x{height}",
            data.len()
        )));
    }
    if width == 0 || height == 0 || out_w == 0 || out_h == 0 {
        return Err(TransformError::ShapeError(format!(
            "PIL bilinear RGB resize {width}x{height} -> {out_w}x{out_h} must be non-empty"
        )));
    }
    let output = resize_pil_bytes::<3>(
        data,
        width,
        height,
        out_w,
        out_h,
        PilResizeFilter::Bilinear,
        KernelQuantization::Pillow,
    );
    RgbImage::from_raw(out_w, out_h, output).ok_or_else(|| {
        TransformError::ShapeError(format!(
            "failed to build PIL bilinear RGB image for {out_w}x{out_h}"
        ))
    })
}

/// Pillow's `Image.resize(.., BICUBIC)` of an `RGBA` image: the colours are
/// premultiplied by their alpha (Pillow's `RGBa` mode, its `MULDIV255` per
/// byte), the four channels go through the same 22-bit kernel as RGB, and the
/// colours are divided back (`255 * v / a`, truncated and clamped; an alpha of
/// 0 or 255 leaves them as they are). An image that keeps its size is copied,
/// as Pillow copies it before any conversion.
pub(crate) fn resize_bicubic_pil_rgba(
    image: &DynamicImage,
    out_w: u32,
    out_h: u32,
) -> image::RgbaImage {
    let owned;
    let rgba = match image {
        DynamicImage::ImageRgba8(rgba) => rgba,
        _ => {
            owned = image.to_rgba8();
            &owned
        }
    };
    let (in_w, in_h) = rgba.dimensions();
    if (in_w, in_h) == (out_w, out_h) {
        return rgba.clone();
    }
    let mut premultiplied = Vec::with_capacity(rgba.as_raw().len());
    for &[r, g, b, a] in rgba.as_raw().as_chunks::<4>().0 {
        premultiplied.extend_from_slice(&[
            mul_div_255(r, a),
            mul_div_255(g, a),
            mul_div_255(b, a),
            a,
        ]);
    }
    let mut output = resize_pil_bytes::<4>(
        &premultiplied,
        in_w,
        in_h,
        out_w,
        out_h,
        PilResizeFilter::Bicubic,
        KernelQuantization::Pillow,
    );
    for pixel in output.as_chunks_mut::<4>().0 {
        let alpha = u32::from(pixel[3]);
        if alpha != 0 && alpha != 255 {
            for value in &mut pixel[..3] {
                *value = (255 * u32::from(*value) / alpha).min(255) as u8;
            }
        }
    }
    #[expect(
        clippy::expect_used,
        reason = "output is exactly out_w*out_h*4 bytes by construction"
    )]
    image::RgbaImage::from_raw(out_w, out_h, output).expect("pil rgba resize buffer size")
}

/// Pillow's `MULDIV255`: `value * alpha / 255` rounded to the nearest byte.
#[inline]
fn mul_div_255(value: u8, alpha: u8) -> u8 {
    let product = u32::from(value) * u32::from(alpha) + 128;
    (((product >> 8) + product) >> 8) as u8
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

/// Pillow-exact BILINEAR resize (RGB8), matching
/// `PIL.Image.resize(.., BILINEAR)`: the triangle filter widened by the
/// downscale factor, Pillow's 22-bit taps and its two passes. This is what
/// torchvision's `resize` computes on a PIL image at its default
/// interpolation, and so what an image processor that resizes the decoded
/// image before converting it to a tensor runs.
pub fn resize_bilinear_pil(image: &DynamicImage, out_w: u32, out_h: u32) -> DynamicImage {
    let rgb = image.to_rgb8();
    let (in_w, in_h) = rgb.dimensions();
    let output = resize_pil_bytes::<3>(
        rgb.as_raw(),
        in_w,
        in_h,
        out_w,
        out_h,
        PilResizeFilter::Bilinear,
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
    } else if in_h == out_h {
        let kernel = FixedKernel::new(in_w, out_w, filter, quantization);
        pil_resample_horizontal::<C>(data, in_h, in_w, out_w, &kernel)
    } else {
        let width = FixedKernel::new(in_w, out_w, filter, quantization);
        let height = FixedKernel::new(in_h, out_h, filter, quantization);
        pil_resample_both::<C>(
            data,
            (in_w, in_h),
            (out_w, out_h),
            &width,
            &height,
            PIL_BAND_BYTES,
        )
    }
}

/// The `window` (`x, y, width, height` of the output) of
/// [`resize_pil_bytes`]'s result, computed alone: the kernels are built for
/// the window's columns and rows ([`FixedKernel::window`]), the width pass
/// runs on those columns over the source rows the window's height taps read
/// (the banded passes read no other), the height pass on those rows, so the
/// bytes are the whole's bytes there and what a crop discards costs nothing.
fn resize_pil_bytes_window<const C: usize>(
    data: &[u8],
    (in_w, in_h): (u32, u32),
    (out_w, out_h): (u32, u32),
    (x, y, w, h): (u32, u32, u32, u32),
    filter: PilResizeFilter,
    quantization: KernelQuantization,
) -> Vec<u8> {
    debug_assert!(x + w <= out_w && y + h <= out_h);
    let (in_w, in_h, out_w, out_h) = (in_w as usize, in_h as usize, out_w as usize, out_h as usize);
    let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let width = || FixedKernel::new(in_w, out_w, filter, quantization).window(x..x + w);
    let height = || FixedKernel::new(in_h, out_h, filter, quantization).window(y..y + h);
    match (in_w != out_w, in_h != out_h) {
        (false, false) => crop_bytes::<C>(data, in_w, (x, y, w, h)),
        // The window's rows are the source's: their columns alone.
        (true, false) => pil_resample_horizontal::<C>(
            &data[y * in_w * C..(y + h) * in_w * C],
            h,
            in_w,
            w,
            &width(),
        ),
        // The window's columns are the source's: their rows alone.
        (false, true) => pil_resample_vertical::<C>(
            &crop_bytes::<C>(data, in_w, (x, 0, w, in_h)),
            h,
            w,
            &height(),
        ),
        (true, true) => pil_resample_both::<C>(
            data,
            (in_w, in_h),
            (w, h),
            &width(),
            &height(),
            PIL_BAND_BYTES,
        ),
    }
}

/// The `(x, y, width, height)` window of a `row_w`-wide buffer of `C`-channel
/// pixels, copied out.
fn crop_bytes<const C: usize>(
    data: &[u8],
    row_w: usize,
    (x, y, w, h): (usize, usize, usize, usize),
) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * C);
    for row in y..y + h {
        out.extend_from_slice(&data[(row * row_w + x) * C..(row * row_w + x + w) * C]);
    }
    out
}

/// `data` (`in_w` x `in_h` interleaved RGB) resampled to `out_w` x `out_h`
/// with `filter`'s kernel in the torch `uint8` quantization, on the passes of
/// this target: on x86-64 the banded Pillow-exact passes of
/// [`resize_bicubic_pil_rgb`] take the torch taps (their AVX-512 and AVX2 arms,
/// or the portable rows), elsewhere the rows of `torch_resample` run them
/// (NEON on AArch64, the plain loops otherwise). One per-sample definition on
/// every route, `clip8((half + Σ pixel · tap) >> precision)`, the width pass
/// rounded to bytes before the height pass, so the bytes are the same.
fn resize_torch_uint8_rgb(
    data: &[u8],
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    filter: PilResizeFilter,
) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        resize_pil_bytes::<3>(
            data,
            in_w,
            in_h,
            out_w,
            out_h,
            filter,
            KernelQuantization::TorchUint8,
        )
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        super::torch_resample::resize_rgb(
            data,
            in_w as usize,
            in_h as usize,
            out_w as usize,
            out_h as usize,
            filter,
        )
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
/// The passes are `resize_torch_uint8_rgb`'s: the width pass rounded to bytes
/// before the height pass, `clip8((half + Σ pixel · tap) >> precision)` per
/// sample.
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
    let output =
        resize_torch_uint8_rgb(data, width, height, out_w, out_h, PilResizeFilter::Bicubic);
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
    Ok(resize_torch_uint8_rgb(
        data,
        width,
        height,
        out_w,
        out_h,
        PilResizeFilter::Lanczos,
    ))
}

/// The torchvision `uint8` BICUBIC resize of [`resize_bicubic_torch_aa_rgb`]
/// for a decoded image.
pub fn resize_bicubic_torch_aa(image: &DynamicImage, out_w: u32, out_h: u32) -> DynamicImage {
    let (in_w, in_h, data) = rgb_bytes(image);
    let output = resize_torch_uint8_rgb(
        &data,
        in_w as u32,
        in_h as u32,
        out_w,
        out_h,
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

/// [`resize_bicubic_torch_aa`] to `out_w` x `out_h` followed by the crop of
/// `crop` (width, height) at `origin` (left, top). On x86-64 the crop's
/// pixels alone, through the window form of the Pillow-exact passes with the
/// torch taps ([`resize_pil_bytes_window`]): the width pass runs on the
/// crop's columns only (a center crop after a shortest-edge resize discards
/// 25 to 44 % of them) and over the source rows the crop's height taps read.
/// Off x86-64, where those passes have their portable rows only, the torch
/// rows ([`resize_torch_uint8_rgb`]: NEON on AArch64) resample the whole and
/// the crop is cut from it. The same bytes either way.
pub(crate) fn resize_bicubic_torch_aa_cropped(
    image: &DynamicImage,
    out_w: u32,
    out_h: u32,
    origin: (u32, u32),
    crop: (u32, u32),
) -> DynamicImage {
    let (in_w, in_h, data) = rgb_bytes(image);
    let output = if cfg!(target_arch = "x86_64") {
        resize_pil_bytes_window::<3>(
            &data,
            (in_w as u32, in_h as u32),
            (out_w, out_h),
            (origin.0, origin.1, crop.0, crop.1),
            PilResizeFilter::Bicubic,
            KernelQuantization::TorchUint8,
        )
    } else {
        let whole = resize_torch_uint8_rgb(
            &data,
            in_w as u32,
            in_h as u32,
            out_w,
            out_h,
            PilResizeFilter::Bicubic,
        );
        crop_bytes::<3>(
            &whole,
            out_w as usize,
            (
                origin.0 as usize,
                origin.1 as usize,
                crop.0 as usize,
                crop.1 as usize,
            ),
        )
    };
    #[expect(
        clippy::expect_used,
        reason = "output is exactly crop.0*crop.1*3 bytes by construction"
    )]
    DynamicImage::ImageRgb8(
        RgbImage::from_raw(crop.0, crop.1, output).expect("torch resize window buffer size"),
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

/// The filters of torch's antialiased float resize, each with the support
/// its CPU kernel widens by the downscale factor.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TorchAaFilter {
    /// The triangle, support 1 (`mode="bilinear"`).
    Bilinear,
    /// Keys' cubic, support 2 (`mode="bicubic"`).
    Bicubic,
}

impl TorchAaFilter {
    fn support(self) -> f32 {
        match self {
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
        }
    }

    /// The filter's weight at `x`, as the kernel evaluates it in `f32`.
    fn weight(self, x: f32) -> f32 {
        match self {
            Self::Bilinear => {
                let x = x.abs();
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Self::Bicubic => torch_aa_cubic_f32(x),
        }
    }
}

/// torch's antialiasing weights of `filter` for one axis, computed as its CPU
/// kernel does on a float tensor: the scale, kernel center and normalized
/// weights in `f32`, and the tap count read from the rounded bounds and
/// capped at the kernel's weight buffer (`ceil(support) * 2 + 1` taps: the
/// rounded bounds can count one more). Pillow's `f64` weights shift by up to
/// ~5e-5 at large output indices, which moves sharp edges by ~0.01.
fn torch_aa_weights(
    in_size: usize,
    out_size: usize,
    filter: TorchAaFilter,
) -> (Vec<(usize, usize)>, Vec<Vec<f32>>) {
    let scale = in_size as f32 / out_size as f32;
    let (support, invscale) = if scale >= 1.0 {
        (filter.support() * scale, 1.0 / scale)
    } else {
        (filter.support(), 1.0)
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
                filter.weight(x as f32)
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
    Ok(torch_aa_resize(
        data,
        in_w,
        in_h,
        out_w,
        out_h,
        TorchAaFilter::Bicubic,
        |_, byte| f32::from(byte),
    ))
}

/// `torch.nn.functional.interpolate(mode="bilinear", antialias=True)` on the
/// image as a float tensor whose values are the bytes mapped through `table`
/// per channel (torchvision's `ToTensor` and `Normalize` as one lookup):
/// interleaved RGB bytes in, interleaved `f32` values out, nothing rounded.
/// The kernel is the triangle widened by the downscale factor, with torch's
/// `f32` weights and its compiled loop's order of roundings
/// (`torch_aa_passes`), so an output is the reference's bit for bit; the
/// width pass runs first, and reads the table instead of a normalized copy of
/// the image.
pub fn resize_bilinear_torch_aa_normalized(
    data: &[u8],
    width: u32,
    height: u32,
    out_w: u32,
    out_h: u32,
    table: &[[f32; 256]; 3],
) -> Result<Vec<f32>> {
    let (in_w, in_h, out_w, out_h) = (
        width as usize,
        height as usize,
        out_w as usize,
        out_h as usize,
    );
    if data.len() != in_w.saturating_mul(in_h).saturating_mul(3) {
        return Err(TransformError::ShapeError(format!(
            "torch bilinear RGB source has {} bytes, expected {} for {width}x{height}",
            data.len(),
            in_w * in_h * 3
        )));
    }
    if in_w == 0 || in_h == 0 || out_w == 0 || out_h == 0 {
        return Err(TransformError::ShapeError(format!(
            "torch bilinear RGB resize {width}x{height} -> {out_w}x{out_h} must be non-empty"
        )));
    }
    Ok(torch_aa_resize(
        data,
        in_w,
        in_h,
        out_w,
        out_h,
        TorchAaFilter::Bilinear,
        |channel, byte| table[channel][usize::from(byte)],
    ))
}

/// The weights of both axes and the passes of torch's antialiased resize over
/// `data` (`in_w` x `in_h` interleaved RGB bytes, each read through `sample`
/// as the float the reference's tensor holds), on the fused multiply-add body
/// where the processor has the instruction.
fn torch_aa_resize(
    data: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    filter: TorchAaFilter,
    sample: impl Fn(usize, u8) -> f32 + Copy,
) -> Vec<f32> {
    let (x_bounds, x_weights) = torch_aa_weights(in_w, out_w, filter);
    let (y_bounds, y_weights) = torch_aa_weights(in_h, out_h, filter);
    let x = TorchAaAxis {
        bounds: &x_bounds,
        weights: &x_weights,
    };
    let y = TorchAaAxis {
        bounds: &y_bounds,
        weights: &y_weights,
    };
    #[cfg(target_arch = "x86_64")]
    if let Some(out) = torch_aa_passes_with_fma(data, in_w, &x, &y, sample) {
        return out;
    }
    torch_aa_passes(data, in_w, &x, &y, sample)
}

/// One axis of the torch-style resize: per output sample the first source
/// index and the tap count, and the taps' weights.
struct TorchAaAxis<'a> {
    bounds: &'a [(usize, usize)],
    weights: &'a [Vec<f32>],
}

/// The two passes of [`resize_bicubic_torch_aa_f32`] and
/// [`resize_bilinear_torch_aa_normalized`]: every source row resampled along
/// the width, then the rows along the height. A source byte enters as
/// `sample(channel, byte)`, the float the reference's tensor holds for it.
/// An output's taps add in the reference's three runs: the first tap is the
/// bare product, the next [`torch_aa_plain_until`] taps add a rounded product
/// in a second rounding, and the rest are fused multiply-adds. Inlined into
/// its callers so that the fused taps compile to the caller's instruction
/// set.
#[inline(always)]
fn torch_aa_passes(
    data: &[u8],
    in_w: usize,
    x: &TorchAaAxis<'_>,
    y: &TorchAaAxis<'_>,
    sample: impl Fn(usize, u8) -> f32 + Copy,
) -> Vec<f32> {
    let (out_w, out_h) = (x.bounds.len(), y.bounds.len());
    let in_h = data.len() / (in_w * 3);

    let mut rows = plain_f32(in_h * out_w * 3);
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
                for (channel, (sum, &byte)) in acc.iter_mut().zip(pixel).enumerate() {
                    *sum = weight * sample(channel, byte);
                }
            }
            for (&weight, pixel) in taps.by_ref().take(torch_aa_plain_until(count)) {
                for (channel, (sum, &byte)) in acc.iter_mut().zip(pixel).enumerate() {
                    *sum += weight * sample(channel, byte);
                }
            }
            for (&weight, pixel) in taps {
                for (channel, (sum, &byte)) in acc.iter_mut().zip(pixel).enumerate() {
                    *sum = weight.mul_add(sample(channel, byte), *sum);
                }
            }
            *out = acc;
        }
    }

    let mut out = plain_f32(out_h * out_w * 3);
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
    sample: impl Fn(usize, u8) -> f32 + Copy,
) -> Vec<f32> {
    torch_aa_passes(data, in_w, x, y, sample)
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
    sample: impl Fn(usize, u8) -> f32 + Copy,
) -> Option<Vec<f32>> {
    if !std::arch::is_x86_feature_detected!("fma") {
        return None;
    }
    // SAFETY: `is_x86_feature_detected!("fma")` answered true just above, the
    // only requirement of a `target_feature(enable = "fma")` function.
    Some(unsafe { torch_aa_passes_fma(data, in_w, x, y, sample) })
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
    if w == h {
        return image.clone();
    }
    // The image centred on the shorter axis of a square of the longer side.
    let side = w.max(h);
    let (x, y) = ((side - w) / 2, (side - h) / 2);
    match image {
        // An opaque image's paste is a copy: the square is built from rows.
        DynamicImage::ImageRgb8(rgb) => {
            DynamicImage::ImageRgb8(paste_rgb(rgb, side, side, x, y, background.0))
        }
        // Any other layout is pasted pixel by pixel, as the generic paste
        // blends it (an alpha channel over the background).
        other => {
            let mut new_image = DynamicImage::from(RgbImage::from_pixel(side, side, background));
            image::imageops::overlay(&mut new_image, other, i64::from(x), i64::from(y));
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
    let resized = RgbImage::from_raw(target_w, target_h, resized).ok_or_else(|| {
        TransformError::ShapeError(format!(
            "failed to build the resized {target_w}x{target_h} image"
        ))
    })?;
    if target_w == out_w && target_h == out_h {
        return Ok(resized);
    }
    // The paste at round((size - resized) * 0.5) along the padded axis, on a
    // canvas of `color`, built from rows.
    let (x, y) = if target_w == out_w {
        (
            0,
            (f64::from(out_h - target_h) * 0.5).round_ties_even() as u32,
        )
    } else {
        (
            (f64::from(out_w - target_w) * 0.5).round_ties_even() as u32,
            0,
        )
    };
    Ok(paste_rgb(&resized, out_w, out_h, x, y, color.0))
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

/// Convert normalized mean values [0, 1] to RGB bytes, truncated like the
/// reference's `int(x * 255)` (the CLIP mean fills with 122, 116, 104).
pub fn mean_to_rgb(mean: &[f64; 3]) -> Rgb<u8> {
    Rgb([
        (mean[0] * 255.0) as u8,
        (mean[1] * 255.0) as u8,
        (mean[2] * 255.0) as u8,
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

// ---------------------------------------------------------------------------
// torch's bicubic interpolation of a float tensor without antialiasing.

/// The four cubic convolution coefficients torch's `upsample_bicubic2d`
/// computes for the fraction `t` (`a = -0.75`, the OpenCV constant; the
/// antialiased kernel uses `-0.5`), in `f32` as its compiled kernel
/// evaluates the two polynomials: `((a + 2) x - (a + 3)) x x + 1` in plain
/// operations, `((a x - 5a) x + 8a) x - 4a` with its first two steps fused
/// multiply-adds and its last one plain. Read off the kernel's coefficients
/// at ratios whose fractions are not dyadic: every other mix of fused and
/// plain steps moves some coefficient by one ULP (the antialiased kernel
/// fuses every step, [`torch_aa_cubic_f32`]).
#[inline]
fn torch_cubic_coefficients(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let near = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let far = |x: f32| A.mul_add(x, -(5.0 * A)).mul_add(x, 8.0 * A) * x - 4.0 * A;
    [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
}

/// One axis of [`resize_bicubic_torch_f32`]: per output index the four
/// clamped source indices and their coefficients, as torch's
/// `compute_indices_weights` lays them out: the source position
/// `scale * (i + 0.5) - 0.5` with `scale = in / out` in `f32` and the
/// product and difference rounded once (the kernel's fused multiply-add;
/// two roundings move the position by one ULP at some indices of a
/// non-dyadic ratio), its floor (capped at the last index) and the
/// fraction above it.
fn torch_cubic_axis(in_size: usize, out_size: usize) -> Vec<([usize; 4], [f32; 4])> {
    let scale = in_size as f32 / out_size as f32;
    let last = in_size as i64 - 1;
    (0..out_size)
        .map(|i| {
            let position = scale.mul_add(i as f32 + 0.5, -0.5);
            let index = (position.floor() as i64).min(last);
            let fraction = (position - index as f32).clamp(0.0, 1.0);
            let taps = std::array::from_fn(|j| (index - 1 + j as i64).clamp(0, last) as usize);
            (taps, torch_cubic_coefficients(fraction))
        })
        .collect()
}

/// Four taps summed as torch's compiled kernel sums them: the second tap's
/// rounded product first, then the first, third and fourth taps as fused
/// multiply-adds (read off the kernel's output on impulse and random inputs;
/// every other order differs in the last bit on some inputs).
#[inline(always)]
fn torch_cubic_dot(taps: [f32; 4], weights: [f32; 4]) -> f32 {
    let sum = taps[1] * weights[1];
    let sum = taps[0].mul_add(weights[0], sum);
    let sum = taps[2].mul_add(weights[2], sum);
    taps[3].mul_add(weights[3], sum)
}

/// The output of [`resize_bicubic_torch_f32`] for every channel: the width
/// dot of each of the four source rows, then the height dot of the four
/// results, the kernel's own nesting. Inlined into its callers so that the
/// fused multiply-adds compile to the caller's instruction set.
#[inline(always)]
fn torch_cubic_passes(
    tensor: &Array3<f32>,
    x_axis: &[([usize; 4], [f32; 4])],
    y_axis: &[([usize; 4], [f32; 4])],
) -> Array3<f32> {
    let channels = tensor.shape()[0];
    let (out_h, out_w) = (y_axis.len(), x_axis.len());
    let mut out = scratch::take_f32(channels * out_h * out_w);
    for (channel, plane) in out.chunks_exact_mut(out_h * out_w).enumerate() {
        let source = tensor.index_axis(ndarray::Axis(0), channel);
        for ((rows, row_weights), out_row) in y_axis.iter().zip(plane.chunks_exact_mut(out_w)) {
            for ((columns, column_weights), value) in x_axis.iter().zip(out_row.iter_mut()) {
                let along_rows =
                    rows.map(|y| torch_cubic_dot(columns.map(|x| source[[y, x]]), *column_weights));
                *value = torch_cubic_dot(along_rows, *row_weights);
            }
        }
    }
    #[expect(
        clippy::expect_used,
        reason = "out has exactly channels*out_h*out_w elements by construction"
    )]
    Array3::from_shape_vec((channels, out_h, out_w), out).expect("shape matches the buffer")
}

/// [`torch_cubic_passes`] compiled for the fused multiply-add instruction:
/// at the baseline x86-64 target the fused taps call the C library's
/// software `fmaf`, the same bits several times slower.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
fn torch_cubic_passes_fma(
    tensor: &Array3<f32>,
    x_axis: &[([usize; 4], [f32; 4])],
    y_axis: &[([usize; 4], [f32; 4])],
) -> Array3<f32> {
    torch_cubic_passes(tensor, x_axis, y_axis)
}

/// `torch.nn.functional.interpolate(mode="bicubic", align_corners=False)` on
/// a float `[C, H, W]` tensor without antialiasing, bit for bit: the cubic
/// convolution with `a = -0.75` on four taps per axis, the taps clamped at
/// the edges, torch's `f32` positions and coefficients and its compiled
/// kernel's order of roundings (`torch_cubic_dot`); verified against the
/// kernel on whole outputs at the integer scales of the tiling processors
/// and at ratios whose fractions are not dyadic (the pinned 5:3, 13:7,
/// 448:336 and 640:336 among them). The Keys kernel of
/// [`bicubic_resize`] (`a = -0.5`, a single summation) differs from it by
/// up to a fifth of the normalized scale on a 2x downscale.
pub fn resize_bicubic_torch_f32(tensor: &Array3<f32>, out_h: usize, out_w: usize) -> Array3<f32> {
    let (h, w) = (tensor.shape()[1], tensor.shape()[2]);
    if out_h == 0 || out_w == 0 || h == 0 || w == 0 {
        return Array3::zeros((tensor.shape()[0], out_h, out_w));
    }
    let x_axis = torch_cubic_axis(w, out_w);
    let y_axis = torch_cubic_axis(h, out_h);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") {
        #[expect(
            unsafe_code,
            reason = "the target-feature function runs only after this function detected its feature"
        )]
        // SAFETY: `is_x86_feature_detected!("fma")` answered true just above,
        // the only requirement of a `target_feature(enable = "fma")` function.
        return unsafe { torch_cubic_passes_fma(tensor, &x_axis, &y_axis) };
    }
    torch_cubic_passes(tensor, &x_axis, &y_axis)
}

#[cfg(test)]
mod tests {

    use image::Rgba;

    use super::*;
    use crate::test_support::{
        assert_same_bytes, create_test_image, gradient_rgb, hex, lcg_bytes, lcg_rgb_image,
    };

    /// Pillow's BILINEAR output on three images of generator bytes (12x5 to
    /// 5x3: a downscale on both axes; 9x4 to 4x4: the width pass alone; 6x7
    /// to 9x10: an upscale), byte for byte: `PIL.Image.resize(size, BILINEAR)`
    /// of Pillow 12.3 on the same bytes.
    #[test]
    fn resize_bilinear_pil_matches_pillow() {
        let expected: [&[u8]; 3] = [
            &[
                147, 178, 85, 89, 108, 108, 140, 116, 110, 145, 127, 168, 73, 90, 118, 112, 156,
                106, 99, 155, 110, 146, 108, 147, 152, 124, 154, 152, 109, 72, 119, 186, 168, 126,
                176, 137, 184, 120, 135, 191, 113, 112, 138, 60, 92,
            ],
            &[
                114, 97, 140, 83, 45, 52, 145, 77, 149, 139, 153, 123, 193, 105, 130, 159, 194,
                120, 108, 90, 131, 87, 141, 209, 120, 203, 71, 60, 160, 173, 92, 161, 173, 53, 151,
                97, 210, 101, 115, 145, 128, 80, 177, 214, 88, 152, 210, 91,
            ],
            &[
                18, 29, 102, 102, 124, 91, 170, 199, 68, 113, 118, 24, 87, 160, 42, 63, 185, 100,
                13, 43, 219, 128, 108, 131, 254, 209, 13, 67, 71, 135, 133, 128, 115, 190, 183, 95,
                153, 167, 98, 118, 144, 102, 99, 115, 120, 125, 83, 181, 175, 126, 140, 219, 177,
                82, 101, 97, 165, 156, 123, 152, 199, 157, 144, 155, 189, 163, 117, 126, 139, 113,
                62, 117, 203, 97, 149, 189, 139, 150, 152, 172, 141, 85, 76, 174, 147, 97, 201,
                180, 121, 216, 69, 140, 175, 45, 115, 103, 72, 77, 62, 165, 45, 147, 118, 135, 157,
                47, 234, 147, 118, 30, 221, 119, 72, 199, 123, 122, 172, 140, 163, 153, 99, 121,
                153, 61, 59, 145, 84, 21, 98, 77, 119, 90, 63, 225, 95, 146, 50, 199, 103, 94, 180,
                80, 147, 162, 165, 190, 169, 131, 134, 187, 75, 69, 175, 71, 74, 64, 82, 140, 81,
                94, 203, 126, 168, 133, 111, 104, 157, 148, 58, 187, 191, 131, 214, 215, 128, 153,
                191, 110, 104, 140, 124, 188, 59, 127, 194, 134, 128, 178, 229, 194, 160, 83, 134,
                136, 149, 78, 133, 195, 90, 219, 116, 93, 148, 98, 98, 69, 115, 119, 125, 181, 134,
                176, 156, 143, 212, 114, 121, 165, 80, 135, 146, 126, 140, 130, 152, 109, 145, 70,
                107, 141, 78, 110, 126, 129, 98, 100, 214, 141, 136, 154, 186, 178, 73, 21, 158,
                88, 122, 166, 99, 208, 149, 99, 151, 53, 57, 141, 134, 96, 131, 211, 156, 74, 97,
                200, 147, 96, 142, 233, 122, 73,
            ],
        ];
        let sizes = [(12, 5, 5, 3, 11), (9, 4, 4, 4, 12), (6, 7, 9, 10, 13)];
        for ((w, h, out_w, out_h, seed), expected) in sizes.into_iter().zip(expected) {
            let out = resize_bilinear_pil(&lcg_rgb_image(w, h, seed), out_w, out_h);
            assert_same_bytes(
                out.as_rgb8().unwrap().as_raw(),
                expected,
                &format!("{w}x{h} -> {out_w}x{out_h}"),
            );
        }
    }

    /// Every byte through `to_tensor_then_normalize` is the value of
    /// torchvision's `ToTensor` then `Normalize` (the three roundings), the
    /// bits of six bytes per channel pinned from torchvision; the fused form
    /// differs from it on some bytes.
    #[test]
    fn to_tensor_then_normalize_takes_torchvisions_roundings() {
        let mean = [0.48145466, 0.4578275, 0.40821073];
        let std = [0.26862954, 0.26130258, 0.27577711];
        let bytes: Vec<u8> = (0..=255u8).flat_map(|byte| [byte; 3]).collect();
        let image = DynamicImage::ImageRgb8(RgbImage::from_raw(16, 16, bytes).unwrap());
        let stepwise = to_tensor_then_normalize(&image, &mean, &std);
        let fused = to_tensor_and_normalize(&image, &mean, &std);
        let pinned: [(usize, [u32; 3]); 6] = [
            (0, [0xbfe5_68dc, 0xbfe0_44b8, 0xbfbd_77d8]),
            (1, [0xbfe3_8a7f, 0xbfde_58f1, 0xbfbb_a5e1]),
            (127, [0x3d7c_e092, 0x3e1d_9531, 0x3ea6_c5e6]),
            (128, [0x3d9c_5620, 0x3e2c_f36b, 0x3eae_0dc4]),
            (254, [0x3ff5_36e5, 0x4003_d502, 0x4008_6d66]),
            (255, [0x3ff7_1541, 0x4004_cae5, 0x4009_5660]),
        ];
        let mut fused_differs = 0;
        for channel in 0..3 {
            let (mean, std) = (mean[channel] as f32, std[channel] as f32);
            for byte in 0..256usize {
                let value = stepwise[[channel, byte / 16, byte % 16]];
                let expected = (byte as f32 / 255.0 - mean) / std;
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "channel {channel} byte {byte}"
                );
                fused_differs += usize::from(fused[[channel, byte / 16, byte % 16]] != value);
            }
            for (byte, bits) in pinned {
                assert_eq!(
                    stepwise[[channel, byte / 16, byte % 16]].to_bits(),
                    bits[channel],
                    "channel {channel} byte {byte}"
                );
            }
        }
        assert!(fused_differs > 0);
    }

    /// torch's non-antialiased bicubic on four generator inputs (values
    /// `byte / 64 - 2`), bit for bit: integer scales (2 and 2; 3 and 1; 4 and
    /// 2) and a fractional one (1.5), the outputs of
    /// `torch.nn.functional.interpolate(mode="bicubic")` of torch 2.13 on the
    /// same values; and a constant input returns its value.
    #[test]
    fn resize_bicubic_torch_f32_matches_torch() {
        let cases: [(usize, usize, u32, [u32; 16]); 4] = [
            (
                8,
                8,
                1,
                [
                    0x3f2f_0300,
                    0x3f32_9000,
                    0xbed9_d600,
                    0x3fd1_8400,
                    0xbf0e_8700,
                    0xbe10_9800,
                    0x3df8_e800,
                    0xbfd1_0000,
                    0xbda5_3000,
                    0xbf9e_2a00,
                    0x3e37_4800,
                    0x3ed8_f000,
                    0xbeb4_2c00,
                    0x3f45_5a00,
                    0x3f30_7c00,
                    0xbf94_1800,
                ],
            ),
            (
                12,
                4,
                2,
                [
                    0xbfbe_0000,
                    0x3fda_0000,
                    0x3fb2_0000,
                    0x3ea8_0000,
                    0xbfca_0000,
                    0x3e98_0000,
                    0x3f7c_0000,
                    0xbff4_0000,
                    0x3f40_0000,
                    0x3ef8_0000,
                    0x3de0_0000,
                    0x3f4c_0000,
                    0xbfaa_0000,
                    0x3f80_0000,
                    0x3e40_0000,
                    0x3f64_0000,
                ],
            ),
            (
                16,
                8,
                3,
                [
                    0x3f11_5e00,
                    0x3ead_a800,
                    0x3e90_8c00,
                    0x3e0a_e400,
                    0xbf80_c600,
                    0xbf3b_6300,
                    0x3f2e_4400,
                    0x3ef5_fe00,
                    0xbf9d_2180,
                    0xbec5_0600,
                    0xbf07_ba00,
                    0xbf0f_8c00,
                    0xbde1_f800,
                    0xbe65_7800,
                    0x3ea5_8000,
                    0xbe0b_b000,
                ],
            ),
            (
                6,
                6,
                4,
                [
                    0xbf87_5316,
                    0x3f90_0138,
                    0xbf36_b784,
                    0xbe4a_5da0,
                    0x3ccb_5e00,
                    0xbf6f_bc5c,
                    0x3f86_a3d4,
                    0x3ebc_7cf8,
                    0xbd94_1360,
                    0xbfc7_2008,
                    0x3f95_db42,
                    0xbf19_857c,
                    0x3f96_e3d4,
                    0x3f21_5c64,
                    0x3faa_ebaa,
                    0xbc98_9e00,
                ],
            ),
        ];
        for (w, h, seed, expected) in cases {
            let values: Vec<f32> = lcg_bytes(w * h, seed)
                .into_iter()
                .map(|byte| f32::from(byte) / 64.0 - 2.0)
                .collect();
            let tensor = Array3::from_shape_vec((1, h, w), values).unwrap();
            let out = resize_bicubic_torch_f32(&tensor, 4, 4);
            assert_eq!(out.shape(), &[1, 4, 4]);
            let bits: Vec<u32> = out.iter().map(|value| value.to_bits()).collect();
            assert_eq!(bits, expected, "{w}x{h} seed {seed}");
        }
        let constant = Array3::from_elem((3, 8, 8), f32::from_bits(0xbfe5_68dc));
        assert!(resize_bicubic_torch_f32(&constant, 4, 4)
            .iter()
            .all(|value| value.to_bits() == 0xbfe5_68dc));
    }

    /// torch's non-antialiased bicubic at ratios whose fractions are not
    /// dyadic, where the kernel's fused multiply-adds (the source position,
    /// the far polynomial's first two steps) and its plain operations (the
    /// near polynomial, the far polynomial's last step) each move a
    /// coefficient by one ULP when swapped: the outputs of torch 2.13 on the
    /// generator inputs, the small ones in full and the large ones as the
    /// SHA-256 of their `f32` bits.
    #[test]
    fn resize_bicubic_torch_f32_matches_torch_at_non_dyadic_ratios() {
        use sha2::{Digest, Sha256};
        let tensor = |w: usize, h: usize, seed: u32| {
            let values: Vec<f32> = lcg_bytes(w * h, seed)
                .into_iter()
                .map(|byte| f32::from(byte) / 64.0 - 2.0)
                .collect();
            Array3::from_shape_vec((1, h, w), values).unwrap()
        };
        // (w, h, out_w, out_h, seed)
        let pinned = [
            (5, 5, 3, 3, 5),
            (10, 10, 3, 3, 14),
            (3, 5, 5, 3, 11),
            (7, 3, 3, 5, 9),
        ];
        let expected: [&[u32]; 4] = [
            &[
                0x3ea6_a17f,
                0xbe89_ecfc,
                0x3eb8_5391,
                0xbf69_ed02,
                0x3ffa_0000,
                0x3eca_f65c,
                0xbf35_d142,
                0x3fc5_4bd5,
                0xbe4b_6f7a,
            ],
            &[
                0xbee9_f3d0,
                0x3e05_b6d3,
                0xbfa5_e18c,
                0xbe09_2a82,
                0x3f3f_2900,
                0x3edf_651e,
                0x3fad_6848,
                0xbfa1_263f,
                0xbfc6_29ee,
            ],
            &[
                0x3dfd_c848,
                0x3f1b_a965,
                0x3f86_25ee,
                0x3eec_bde2,
                0xbde0_3cd8,
                0x3ffd_8d52,
                0x3fc3_72a8,
                0x3f18_0000,
                0xbf58_7ef9,
                0xbfe9_1270,
                0xbfb2_3815,
                0xbfe9_0b84,
                0xbfe2_097b,
                0x3ee5_6773,
                0x400f_8c08,
            ],
            &[
                0x3f81_8494,
                0x3f09_0e56,
                0x3fbe_2b7b,
                0x3e56_75f9,
                0x3e14_fe04,
                0x3ee2_b421,
                0xbf43_ed0f,
                0xbd40_0000,
                0xbf0f_a128,
                0xbf3b_81dc,
                0x3f85_893a,
                0x3ef9_193c,
                0xbefe_a62b,
                0x3ffc_a7f2,
                0x3fc7_240d,
            ],
        ];
        for ((w, h, out_w, out_h, seed), expected) in pinned.into_iter().zip(expected) {
            let out = resize_bicubic_torch_f32(&tensor(w, h, seed), out_h, out_w);
            let bits: Vec<u32> = out.iter().map(|value| value.to_bits()).collect();
            assert_eq!(bits, expected, "{w}x{h} -> {out_w}x{out_h} seed {seed}");
        }
        let digested = [
            (13, 13, 7, 7, 6),
            (9, 7, 7, 5, 10),
            (4, 4, 7, 7, 12),
            (448, 448, 336, 336, 8),
            (640, 640, 336, 336, 7),
            (1008, 1008, 336, 336, 13),
        ];
        let digests = [
            "36a2bd819c84946a5b479acba6af42ad0387c20a817599ae25c1588a6e7e18dd",
            "fef907498d1618ff67edfbaad3eb6838de577fab47f31e20940a901ac30a6cf5",
            "a1d36322c57c03b75c5089da73755c6ba237811cf6392ee9fbc5498bffff1f3c",
            "8a3eba7b7189cadb42453f76787cd7f77c9b310a4313239f78c3896dd54baf19",
            "7eafa75bda050c73f8cd85ef6cfb1e2eceef901b94b98de065ff5cace220e2c8",
            "a18747fb9c63193e811aa291531f8b9a408d3f461cc8bd72e73c9849b8f8a04c",
        ];
        for ((w, h, out_w, out_h, seed), digest) in digested.into_iter().zip(digests) {
            let out = resize_bicubic_torch_f32(&tensor(w, h, seed), out_h, out_w);
            let bytes: Vec<u8> = out
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect();
            let actual = hex(&Sha256::digest(&bytes));
            assert_eq!(actual, digest, "{w}x{h} -> {out_w}x{out_h} seed {seed}");
        }
    }

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

    /// Both tap quantizations the passes run: Pillow's fixed 22 bits and the
    /// torch `uint8` resampler's 16-bit taps.
    const QUANTIZATIONS: [KernelQuantization; 2] =
        [KernelQuantization::Pillow, KernelQuantization::TorchUint8];

    /// The passes' parity cases: up- and downscales, identity axes, extreme
    /// aspects, row lengths that leave vector tails, odd row counts (the
    /// vector width pass's lone last row), a large image, and the vector
    /// passes' edges (an axis shorter than one padded window, windows of
    /// eight, nine and seven taps, so a tap paired with a zero and a window
    /// of one tap past a multiple of four, a downscale by 154 whose padded
    /// windows run hundreds of source pixels).
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
    /// Lanczos, both tap quantizations, over [`PIL_PASS_CASES`], equals the
    /// per-sample `i64` computation.
    #[test]
    fn pil_passes_equal_the_per_sample_i64_reference() {
        fn check<const C: usize>(
            data: &[u8],
            (in_w, in_h, out_w, out_h): (usize, usize, usize, usize),
            filter: PilResizeFilter,
            quantization: KernelQuantization,
            what: &str,
        ) {
            let actual = resize_pil_bytes::<C>(
                data,
                in_w as u32,
                in_h as u32,
                out_w as u32,
                out_h as u32,
                filter,
                quantization,
            );
            let expected = per_sample_reference_channels::<C>(
                data,
                in_w,
                in_h,
                out_w,
                out_h,
                filter,
                quantization,
            );
            assert_same_bytes(&actual, &expected, &format!("{C} channels, {what}"));
        }
        for quantization in QUANTIZATIONS {
            for (f, filter) in [
                PilResizeFilter::Bicubic,
                PilResizeFilter::Lanczos,
                PilResizeFilter::Bilinear,
            ]
            .into_iter()
            .enumerate()
            {
                for (seed, &case) in PIL_PASS_CASES.iter().enumerate() {
                    let (in_w, in_h, out_w, out_h) = case;
                    let seed = seed * 13 + f * 101;
                    let what = format!(
                        "{quantization:?} taps, filter {f}: {in_w}x{in_h} -> {out_w}x{out_h}"
                    );
                    check::<3>(
                        &gradient_rgb(in_w, in_h, seed),
                        case,
                        filter,
                        quantization,
                        &what,
                    );
                    check::<4>(
                        &gradient_rgba(in_w, in_h, seed),
                        case,
                        filter,
                        quantization,
                        &what,
                    );
                }
            }
        }
    }

    /// A window of the resample computed alone equals the crop of the whole,
    /// under both tap quantizations: the LLaVA 1.5 shapes (a 16:9 and a 9:16
    /// source to a 336 short side, the center crop's window), small windows
    /// at the edges, an axis kept, both axes kept.
    #[test]
    fn a_window_of_the_resample_equals_the_crop_of_the_whole() {
        let cases = [
            (1920, 1080, 597, 336, (130, 0, 336, 336)),
            (1080, 1920, 336, 597, (0, 130, 336, 336)),
            (33, 17, 7, 5, (0, 0, 3, 2)),
            (33, 17, 7, 5, (4, 3, 3, 2)),
            (7, 5, 33, 17, (30, 15, 3, 2)),
            (31, 23, 31, 17, (5, 4, 20, 9)),
            (31, 23, 19, 23, (2, 20, 10, 3)),
            (31, 23, 31, 23, (3, 2, 10, 5)),
        ];
        for quantization in QUANTIZATIONS {
            for (seed, &(in_w, in_h, out_w, out_h, (x, y, w, h))) in cases.iter().enumerate() {
                let data = gradient_rgb(in_w, in_h, seed * 7 + 3);
                let whole = resize_pil_bytes::<3>(
                    &data,
                    in_w as u32,
                    in_h as u32,
                    out_w as u32,
                    out_h as u32,
                    PilResizeFilter::Bicubic,
                    quantization,
                );
                let actual = resize_pil_bytes_window::<3>(
                    &data,
                    (in_w as u32, in_h as u32),
                    (out_w as u32, out_h as u32),
                    (x as u32, y as u32, w as u32, h as u32),
                    PilResizeFilter::Bicubic,
                    quantization,
                );
                assert_same_bytes(
                    &actual,
                    &crop_bytes::<3>(&whole, out_w, (x, y, w, h)),
                    &format!(
                        "{quantization:?} taps, {in_w}x{in_h} -> {out_w}x{out_h}, window {x},{y} {w}x{h}"
                    ),
                );
            }
        }
    }

    /// The cropped entry point equals the resize followed by the crop on
    /// whichever route this target takes (the window form on x86-64, the
    /// torch rows and a crop elsewhere): the LLaVA 1.5 pair.
    #[test]
    fn the_cropped_torch_resize_equals_the_resize_and_the_crop() {
        let cases = [
            (1920, 1080, 597, 336, (130, 0, 336, 336)),
            (1080, 1920, 336, 597, (0, 130, 336, 336)),
        ];
        for (seed, &(in_w, in_h, out_w, out_h, (x, y, w, h))) in cases.iter().enumerate() {
            let data = gradient_rgb(in_w as usize, in_h as usize, seed * 11 + 5);
            let image = DynamicImage::ImageRgb8(RgbImage::from_raw(in_w, in_h, data).unwrap());
            let whole = resize_bicubic_torch_aa(&image, out_w, out_h).into_rgb8();
            assert_same_bytes(
                resize_bicubic_torch_aa_cropped(&image, out_w, out_h, (x, y), (w, h))
                    .into_rgb8()
                    .as_raw(),
                crop_rgb(&whole, x, y, w, h).as_raw(),
                &format!("{in_w}x{in_h} -> {out_w}x{out_h}, crop {w}x{h} at {x},{y}"),
            );
        }
    }

    /// The torch `uint8` entry points carry the per-sample bytes of the torch
    /// taps on every case, whichever rows this target runs them on (the
    /// Pillow-exact passes on x86-64, the torch module's rows elsewhere), and
    /// the decoded-image form equals the byte form.
    #[test]
    fn torch_uint8_entry_points_carry_the_per_sample_bytes() {
        for (seed, &(in_w, in_h, out_w, out_h)) in PIL_PASS_CASES.iter().enumerate() {
            let data = gradient_rgb(in_w, in_h, seed * 19 + 7);
            let (w, h, ow, oh) = (in_w as u32, in_h as u32, out_w as u32, out_h as u32);
            let what = format!("{in_w}x{in_h} -> {out_w}x{out_h}");
            let bicubic = resize_bicubic_torch_aa_rgb(&data, w, h, ow, oh).unwrap();
            assert_same_bytes(
                bicubic.as_raw(),
                &per_sample_reference_channels::<3>(
                    &data,
                    in_w,
                    in_h,
                    out_w,
                    out_h,
                    PilResizeFilter::Bicubic,
                    KernelQuantization::TorchUint8,
                ),
                &format!("torch bicubic, {what}"),
            );
            let image = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, data.clone()).unwrap());
            assert_same_bytes(
                resize_bicubic_torch_aa(&image, ow, oh).to_rgb8().as_raw(),
                bicubic.as_raw(),
                &format!("torch bicubic of the decoded image, {what}"),
            );
            assert_same_bytes(
                &resize_lanczos_torch_aa_rgb_bytes(&data, w, h, ow, oh).unwrap(),
                &per_sample_reference_channels::<3>(
                    &data,
                    in_w,
                    in_h,
                    out_w,
                    out_h,
                    PilResizeFilter::Lanczos,
                    KernelQuantization::TorchUint8,
                ),
                &format!("torch lanczos, {what}"),
            );
        }
    }

    /// The portable arm and every vector arm the host can build (the
    /// detected one, and on x86-64 each of AVX2 and AVX-512 the processor
    /// reports) write the same bytes on every case, three and four channels,
    /// both tap quantizations (on a processor without a vector unit every arm
    /// is the portable loop).
    #[test]
    fn pil_pass_arms_agree() {
        fn check<const C: usize>(
            data: &[u8],
            (in_w, in_h, out_w, out_h): (usize, usize, usize, usize),
            filter: PilResizeFilter,
            quantization: KernelQuantization,
            what: &str,
        ) {
            let kernel = FixedKernel::new(in_w, out_w, filter, quantization);
            let mut portable = vec![0_u8; in_h * out_w * C];
            WidthPass::Portable.rows::<C>(
                data,
                &kernel,
                in_w,
                out_w,
                0,
                &mut portable,
                &mut BandScratch::default(),
            );
            #[cfg_attr(
                not(target_arch = "x86_64"),
                expect(unused_mut, reason = "only the x86-64 arms are pushed")
            )]
            let mut arms = vec![("detected", WidthPass::new::<C>(&kernel))];
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    let split = SplitTaps::new(&kernel, 0, kernel.bounds.len(), 4);
                    arms.push(("avx2", WidthPass::Avx2(split)));
                }
                if avx512::detected() {
                    if let Some(groups) = GroupTaps::new::<C>(&kernel) {
                        arms.push(("avx512", WidthPass::Avx512(groups)));
                    }
                }
            }
            for (name, arm) in &arms {
                let mut detected = vec![0_u8; in_h * out_w * C];
                arm.rows::<C>(
                    data,
                    &kernel,
                    in_w,
                    out_w,
                    0,
                    &mut detected,
                    &mut BandScratch::default(),
                );
                assert_same_bytes(
                    &detected,
                    &portable,
                    &format!("width pass {name}, {C} channels, {what}"),
                );
            }
            let kernel = FixedKernel::new(in_h, out_h, filter, quantization);
            let mut portable_rows = vec![0_u8; out_h * out_w * C];
            HeightPass::Portable.rows(&portable, 0, &kernel, out_w * C, 0, &mut portable_rows);
            #[cfg_attr(
                not(target_arch = "x86_64"),
                expect(unused_mut, reason = "only the x86-64 arms are pushed")
            )]
            let mut arms = vec![("detected", HeightPass::new())];
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    arms.push(("avx2", HeightPass::Avx2));
                }
                if avx512::detected() {
                    arms.push(("avx512", HeightPass::Avx512));
                }
            }
            for (name, arm) in &arms {
                let mut detected_rows = vec![0_u8; out_h * out_w * C];
                arm.rows(&portable, 0, &kernel, out_w * C, 0, &mut detected_rows);
                assert_same_bytes(
                    &detected_rows,
                    &portable_rows,
                    &format!("height pass {name}, {C} channels, {what}"),
                );
            }
        }
        for quantization in QUANTIZATIONS {
            for (f, filter) in [
                PilResizeFilter::Bicubic,
                PilResizeFilter::Lanczos,
                PilResizeFilter::Bilinear,
            ]
            .into_iter()
            .enumerate()
            {
                for (seed, &case) in PIL_PASS_CASES.iter().enumerate() {
                    let (in_w, in_h, out_w, out_h) = case;
                    let what = format!(
                        "{quantization:?} taps, filter {f}: {in_w}x{in_h} -> {out_w}x{out_h}"
                    );
                    check::<3>(
                        &gradient_rgb(in_w, in_h, seed),
                        case,
                        filter,
                        quantization,
                        &what,
                    );
                    check::<4>(
                        &gradient_rgba(in_w, in_h, seed),
                        case,
                        filter,
                        quantization,
                        &what,
                    );
                }
            }
        }
    }

    /// The banded passes write the per-sample reference's bytes whatever the
    /// band budget: one output row per band (every band carries most of its
    /// rows from the one before), a few rows, the shipped budget and one
    /// band for the whole image; the cases cover downscales, upscales, one
    /// axis each way and a row whose windows alone exceed the budget.
    #[test]
    fn pil_banded_passes_carry_rows_between_bands() {
        fn check<const C: usize>(
            data: &[u8],
            (in_w, in_h, out_w, out_h): (usize, usize, usize, usize),
            filter: PilResizeFilter,
            quantization: KernelQuantization,
            what: &str,
        ) {
            let width = FixedKernel::new(in_w, out_w, filter, quantization);
            let height = FixedKernel::new(in_h, out_h, filter, quantization);
            let expected = per_sample_reference_channels::<C>(
                data,
                in_w,
                in_h,
                out_w,
                out_h,
                filter,
                quantization,
            );
            let row_len = out_w * C;
            for band_bytes in [
                0,
                row_len,
                3 * row_len,
                17 * row_len,
                PIL_BAND_BYTES,
                usize::MAX,
            ] {
                let actual = pil_resample_both::<C>(
                    data,
                    (in_w, in_h),
                    (out_w, out_h),
                    &width,
                    &height,
                    band_bytes,
                );
                assert_same_bytes(
                    &actual,
                    &expected,
                    &format!("{C} channels, {what}, bands of {band_bytes} bytes"),
                );
            }
        }
        for quantization in QUANTIZATIONS {
            for (f, filter) in [
                PilResizeFilter::Bicubic,
                PilResizeFilter::Lanczos,
                PilResizeFilter::Bilinear,
            ]
            .into_iter()
            .enumerate()
            {
                for (seed, &case) in PIL_PASS_CASES.iter().enumerate() {
                    let (in_w, in_h, out_w, out_h) = case;
                    if in_w == out_w || in_h == out_h {
                        continue;
                    }
                    let seed = seed * 17 + f * 103;
                    let what = format!(
                        "{quantization:?} taps, filter {f}: {in_w}x{in_h} -> {out_w}x{out_h}"
                    );
                    check::<3>(
                        &gradient_rgb(in_w, in_h, seed),
                        case,
                        filter,
                        quantization,
                        &what,
                    );
                    check::<4>(
                        &gradient_rgba(in_w, in_h, seed),
                        case,
                        filter,
                        quantization,
                        &what,
                    );
                }
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

    /// The split the 16-bit vector arms rest on, checked over the same sweep
    /// for both filters: every tap is `hi · 2^11 + lo` with `lo` in
    /// `[-2^10, 2^10)` and both halves inside a 16-bit lane (so a
    /// multiply-add's pair of products is far inside `i32`), and over any
    /// output's window each partial sum stays inside `i32` for every pixel
    /// assignment: a sum of `pixel · coefficient` over pixels in `0..=255`
    /// is largest at 255 under the positive coefficients and smallest at 255
    /// under the negative ones, so the high half's partial sum shifted by
    /// eleven and the low half's with the rounding term are bounded by sign.
    #[test]
    fn pil_split_taps_fit_the_16_bit_lanes_and_32_bit_sums() {
        let sizes = [1usize, 2, 3, 5, 7, 16, 100, 1000, 4096];
        let (max, min) = (i64::from(i32::MAX), i64::from(i32::MIN));
        for filter in [PilResizeFilter::Bicubic, PilResizeFilter::Lanczos] {
            for &in_size in &sizes {
                for &out_size in &sizes {
                    let k = FixedKernel::new(in_size, out_size, filter, KernelQuantization::Pillow);
                    let half = 1_i64 << (k.precision - 1);
                    for taps in &k.taps {
                        // The positive and the negative coefficient sums of each half.
                        let (mut hi_up, mut hi_down, mut lo_up, mut lo_down) =
                            (0_i64, 0_i64, 0_i64, 0_i64);
                        for &tap in taps {
                            let (hi, lo) = split_tap(tap);
                            assert_eq!(
                                (i32::from(hi) << SPLIT_LOW_BITS) + i32::from(lo),
                                tap,
                                "{in_size} -> {out_size}: tap {tap} is not its halves recombined"
                            );
                            assert!(
                                (-(1 << (SPLIT_LOW_BITS - 1))..1 << (SPLIT_LOW_BITS - 1)).contains(&lo),
                                "{in_size} -> {out_size}: low half {lo} of tap {tap} outside its bits"
                            );
                            let (hi, lo) = (i64::from(hi), i64::from(lo));
                            hi_up += hi.max(0);
                            hi_down += hi.min(0);
                            lo_up += lo.max(0);
                            lo_down += lo.min(0);
                        }
                        let what = format!("{in_size} -> {out_size}: Σ hi {hi_up} / {hi_down}, Σ lo {lo_up} / {lo_down}");
                        assert!(
                            (255 * hi_up) << SPLIT_LOW_BITS <= max
                                && (255 * hi_down) << SPLIT_LOW_BITS >= min,
                            "{what}: the shifted high partial sum leaves i32"
                        );
                        assert!(
                            half + 255 * lo_up <= max && half + 255 * lo_down >= min,
                            "{what}: the low partial sum leaves i32"
                        );
                    }
                }
            }
        }
    }

    /// The byte chunks the AVX-512 arms rest on, checked over the same sweep
    /// for both filters: every tap is `c2 · 2^16 + c1 · 2^8 + c0` with each
    /// chunk a signed byte, and over any output's window every stage of the
    /// arms' sums stays inside `i32` for every pixel assignment: each level's
    /// own sum, the top level shifted and added to the middle one, that
    /// shifted again, and the whole with the rounding term; a sum of `pixel ·
    /// coefficient` over pixels in `0..=255` is largest at 255 under the
    /// positive coefficients and smallest at 255 under the negative ones.
    #[test]
    fn pil_tap_chunks_fit_the_bytes_and_32_bit_sums() {
        let sizes = [1usize, 2, 3, 5, 7, 16, 100, 1000, 4096];
        let (max, min) = (i64::from(i32::MAX), i64::from(i32::MIN));
        for filter in [PilResizeFilter::Bicubic, PilResizeFilter::Lanczos] {
            for &in_size in &sizes {
                for &out_size in &sizes {
                    let k = FixedKernel::new(in_size, out_size, filter, KernelQuantization::Pillow);
                    let half = 1_i64 << (k.precision - 1);
                    for taps in &k.taps {
                        // Per stage, the positive and the negative coefficient sums.
                        let (mut up, mut down) = ([0_i64; 6], [0_i64; 6]);
                        for &tap in taps {
                            let [c0, c1, c2] = tap_chunks(tap);
                            assert_eq!(
                                (i32::from(c2) << 16) + (i32::from(c1) << 8) + i32::from(c0),
                                tap,
                                "{in_size} -> {out_size}: tap {tap} is not its chunks recombined"
                            );
                            let (c0, c1, c2) = (i64::from(c0), i64::from(c1), i64::from(c2));
                            let high = (c2 << CHUNK_BITS) + c1;
                            let stages = [c0, c1, c2, high, high << CHUNK_BITS, i64::from(tap)];
                            for ((up, down), coefficient) in
                                up.iter_mut().zip(&mut down).zip(stages)
                            {
                                *up += coefficient.max(0);
                                *down += coefficient.min(0);
                            }
                        }
                        for (stage, (&up, &down)) in up.iter().zip(&down).enumerate() {
                            let rounding = if stage == 0 || stage == 5 { half } else { 0 };
                            assert!(
                                rounding + 255 * up <= max && rounding + 255 * down >= min,
                                "{in_size} -> {out_size}: stage {stage} of the chunk sums leaves i32 (Σ {up} / {down})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The AVX-512 width arm takes an axis only while its output groups
    /// gather their four-tap windows from one 128-byte window: five RGB
    /// outputs per vector (four RGBA) on the usual downscales, fewer as the
    /// scale grows, none past about thirty, where the axis keeps the
    /// split-tap arm.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pil_group_taps_take_only_windows_that_fit_the_gather() {
        let groups = |in_size, out_size| {
            let kernel = FixedKernel::new(
                in_size,
                out_size,
                PilResizeFilter::Bicubic,
                KernelQuantization::Pillow,
            );
            (
                GroupTaps::new::<3>(&kernel).map(|groups| groups.outputs),
                GroupTaps::new::<4>(&kernel).map(|groups| groups.outputs),
            )
        };
        assert_eq!(groups(8192, 3584), (Some(5), Some(4)), "a 2.3x downscale");
        assert_eq!(groups(8192, 1344), (Some(5), Some(4)), "a 6.1x downscale");
        assert_eq!(groups(1000, 100), (Some(4), Some(3)), "a 10x downscale");
        assert_eq!(groups(64, 2000), (Some(5), Some(4)), "an upscale");
        assert_eq!(groups(2000, 13), (None, None), "a 154x downscale");
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

    /// A source with no pixels is refused before the passes run (its byte
    /// count matches an empty buffer, so the length check lets it through);
    /// the banded passes would otherwise take their row count from it.
    #[test]
    fn resize_bicubic_pil_rgb_rejects_an_empty_source() {
        for (width, height) in [(0, 4), (4, 0), (0, 0)] {
            assert!(matches!(
                resize_bicubic_pil_rgb(&[], width, height, 2, 2),
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
    fn expand_to_square_is_the_generic_paste() {
        // A wide image (padded above and below), a tall one (padded on the
        // sides), and odd sizes whose padding does not split evenly.
        for (w, h) in [(100u32, 50u32), (50, 100), (31, 17), (17, 31)] {
            let image = DynamicImage::ImageRgb8(
                RgbImage::from_raw(w, h, gradient_rgb(w as usize, h as usize, 9))
                    .expect("gradient size"),
            );
            let background = Rgb([200, 30, 90]);
            let side = w.max(h);
            let mut expected = DynamicImage::from(RgbImage::from_pixel(side, side, background));
            image::imageops::overlay(
                &mut expected,
                &image,
                i64::from((side - w) / 2),
                i64::from((side - h) / 2),
            );
            let squared = expand_to_square(&image, background);
            assert_eq!(squared.dimensions(), (side, side), "{w}x{h}");
            assert_eq!(squared.as_bytes(), expected.as_bytes(), "{w}x{h}");
        }
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
        // Truncated, as the reference's `int(x * 255)`.
        assert_eq!(mean_to_rgb(&[0.5, 0.25, 1.0]), Rgb([127, 63, 255]));
        assert_eq!(
            mean_to_rgb(&[0.48145466, 0.4578275, 0.40821073]),
            Rgb([122, 116, 104])
        );
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
            let (x_bounds, x_weights) =
                torch_aa_weights(width as usize, out_w as usize, TorchAaFilter::Bicubic);
            let (y_bounds, y_weights) =
                torch_aa_weights(height as usize, out_h as usize, TorchAaFilter::Bicubic);
            let x = TorchAaAxis {
                bounds: &x_bounds,
                weights: &x_weights,
            };
            let y = TorchAaAxis {
                bounds: &y_bounds,
                weights: &y_weights,
            };
            let bytes_as_floats = |_: usize, byte: u8| f32::from(byte);
            #[cfg(target_arch = "x86_64")]
            let fma = torch_aa_passes_with_fma(&data, width as usize, &x, &y, bytes_as_floats);
            #[cfg(not(target_arch = "x86_64"))]
            let fma: Option<Vec<f32>> = None;
            let bodies = [
                (
                    "the dispatching entry point",
                    Some(resize_bicubic_torch_aa_f32(&data, width, height, out_w, out_h).unwrap()),
                ),
                (
                    "the generic body",
                    Some(torch_aa_passes(
                        &data,
                        width as usize,
                        &x,
                        &y,
                        bytes_as_floats,
                    )),
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

    /// [`resize_bilinear_torch_aa_normalized`] carries torch's bits: eight
    /// seeded inputs (an LCG's bytes) through torchvision's `ToTensor` and
    /// `Normalize` (CLIP's constants) and `interpolate(mode="bilinear",
    /// antialias=True)` on the CPU (torch 2.13.0), one- to three-tap
    /// upscales, a two-, a three- and a nine-tap downscale, a mixed resize
    /// and the 728 and 504 targets of the processor that reads it: the
    /// digest of every output's bits and four elements per case. The
    /// dispatching entry point and the generic body are each driven (the
    /// entry point takes the FMA body where the processor has it).
    #[test]
    fn resize_bilinear_torch_aa_normalized_carries_the_reference_bits() {
        use sha2::{Digest, Sha256};

        use crate::vision::processors::llava::{CLIP_MEAN, CLIP_STD};

        struct Case {
            width: u32,
            height: u32,
            out_w: u32,
            out_h: u32,
            seed: u32,
            // SHA-256 of the little-endian f32 bits of the reference's output.
            digest: &'static str,
            // (index, bits) of four elements of the output.
            witnesses: [(usize, u32); 4],
        }

        let cases = [
            Case {
                width: 5,
                height: 7,
                out_w: 3,
                out_h: 4,
                seed: 11,
                digest: "1ad580f42b16fbfb95c1688910bb6e113f6fcc0e5c1cd849c0c7effbf169007f",
                witnesses: [
                    (0, 0x3da4_acce),
                    (1, 0x3f5f_2862),
                    (18, 0xbee5_2ec1),
                    (35, 0x3db8_af89),
                ],
            },
            Case {
                width: 13,
                height: 1,
                out_w: 7,
                out_h: 1,
                seed: 23,
                digest: "bcecf36514b113c486095279637004115fe5f1d65a438c50e509a9ae7653e817",
                witnesses: [
                    (0, 0x3fb1_f1e4),
                    (1, 0xbf20_9217),
                    (10, 0xbf2d_7353),
                    (20, 0xbf6a_e20a),
                ],
            },
            Case {
                width: 33,
                height: 3,
                out_w: 8,
                out_h: 3,
                seed: 37,
                digest: "5eedd24315d20575ad03f6072e6ff4627dff8e21f252e7445d3dd0f6119af767",
                witnesses: [
                    (0, 0x3dd8_e297),
                    (1, 0x3db4_fc90),
                    (36, 0x3e8c_1590),
                    (71, 0x3f6e_548f),
                ],
            },
            Case {
                width: 11,
                height: 5,
                out_w: 12,
                out_h: 8,
                seed: 5,
                digest: "590b83e537a4e8e8749b891c5dd6c17efe27b432c828dbb804ac0c387e3e46f6",
                witnesses: [
                    (0, 0x3fbb_49b2),
                    (1, 0xbfce_fabf),
                    (144, 0xbf3a_00f9),
                    (287, 0xbeb6_b304),
                ],
            },
            Case {
                width: 41,
                height: 29,
                out_w: 728,
                out_h: 728,
                seed: 19,
                digest: "833a98c20a53ba6010c32eec38290280434e90fcd49943a4c7b4c0581585e4ef",
                witnesses: [
                    (0, 0x3f64_d338),
                    (1, 0x3fe7_01d9),
                    (794_976, 0xbfae_fd7d),
                    (1_589_951, 0xbe5b_c8fb),
                ],
            },
            Case {
                width: 97,
                height: 61,
                out_w: 504,
                out_h: 504,
                seed: 41,
                digest: "6c8f02a814ebcf870027b81cbbf295f15496edb0a02072d39cb37e9e467a1a1d",
                witnesses: [
                    (0, 0x3f85_1939),
                    (1, 0x3fd9_8f6d),
                    (381_024, 0xbfcc_472c),
                    (762_047, 0xbf9c_b482),
                ],
            },
            Case {
                width: 1030,
                height: 9,
                out_w: 728,
                out_h: 7,
                seed: 43,
                digest: "b7c7db000d20ad32fca3ec102697a09d3b32bd2d76217e4a583dc08cfd7798f6",
                witnesses: [
                    (0, 0xbebd_7b58),
                    (1, 0xbd1b_d6a9),
                    (7644, 0x3f36_7cc1),
                    (15287, 0xbe59_8f71),
                ],
            },
            Case {
                width: 2,
                height: 3,
                out_w: 504,
                out_h: 504,
                seed: 47,
                digest: "7801aba2fd2ce9afaafca0d1d7f23a3e49a4ecd132cf77e90c3f83ca4738d9c3",
                witnesses: [
                    (0, 0xbe8c_4d29),
                    (1, 0x3f7a_b480),
                    (381_024, 0xbd51_a53b),
                    (762_047, 0x3d29_46d1),
                ],
            },
        ];
        let table = to_tensor_then_normalize_table(&CLIP_MEAN, &CLIP_STD);
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
            let (x_bounds, x_weights) =
                torch_aa_weights(width as usize, out_w as usize, TorchAaFilter::Bilinear);
            let (y_bounds, y_weights) =
                torch_aa_weights(height as usize, out_h as usize, TorchAaFilter::Bilinear);
            let x = TorchAaAxis {
                bounds: &x_bounds,
                weights: &x_weights,
            };
            let y = TorchAaAxis {
                bounds: &y_bounds,
                weights: &y_weights,
            };
            let bodies = [
                (
                    "the dispatching entry point",
                    resize_bilinear_torch_aa_normalized(&data, width, height, out_w, out_h, &table)
                        .unwrap(),
                ),
                (
                    "the generic body",
                    torch_aa_passes(&data, width as usize, &x, &y, |channel, byte| {
                        table[channel][usize::from(byte)]
                    }),
                ),
            ];
            for (body, out) in bodies {
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
    /// `RGBA` bytes from a 32-bit linear congruential generator (the top byte
    /// of the state per byte), a third of the pixels fully transparent, a
    /// third opaque, the rest at the generator's alpha.
    fn generated_rgba(width: u32, height: u32, seed: u32) -> image::RgbaImage {
        let mut state = seed;
        let mut bytes: Vec<u8> = (0..width * height * 4)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        for (index, pixel) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            match index % 3 {
                0 => pixel[3] = 0,
                1 => pixel[3] = 255,
                _ => {}
            }
        }
        image::RgbaImage::from_raw(width, height, bytes).unwrap()
    }

    /// Pillow 12.3.0's `Image.resize(BICUBIC)` of two generated `RGBA` images,
    /// a downscale and an upscale: the SHA-256 of the output bytes, with the
    /// first eight bytes in the clear. Resampling the four channels without
    /// the premultiplication differs on 72 of the 96 and on 208 of the 288
    /// bytes.
    #[test]
    fn resize_bicubic_pil_rgba_carries_pillows_bytes() {
        use sha2::{Digest, Sha256};

        for (width, height, out_w, out_h, seed, digest, first_eight) in [
            (
                11,
                7,
                6,
                4,
                3,
                "1c242764fe95e780c2d12292983746425d9ca5c0d6470d688034ed0f0cee921d",
                [136, 153, 112, 118, 23, 188, 106, 153],
            ),
            (
                5,
                3,
                9,
                8,
                5,
                "8d440c419d8017bee119cb4eda912e9b5cbded0311a8783746e9b072780dc7d7",
                [0, 0, 0, 0, 176, 88, 119, 75],
            ),
        ] {
            let image = DynamicImage::ImageRgba8(generated_rgba(width, height, seed));
            let out = resize_bicubic_pil_rgba(&image, out_w, out_h);
            assert_eq!(
                out.as_raw()[..8],
                first_eight,
                "{width}x{height} -> {out_w}x{out_h}: the first eight bytes"
            );
            let actual = hex(&Sha256::digest(out.as_raw()));
            assert_eq!(
                actual, digest,
                "{width}x{height} -> {out_w}x{out_h}: the digest"
            );
        }
    }

    /// An opaque `RGBA` image resizes to the RGB kernel's bytes with its alpha
    /// kept at 255, and an image that keeps its size comes back as it is: no
    /// premultiplication round trip, which a half-transparent pixel would not
    /// survive.
    #[test]
    fn resize_bicubic_pil_rgba_agrees_with_the_rgb_kernel_when_opaque_and_copies_an_unchanged_size()
    {
        let (width, height) = (9, 6);
        let mut opaque = generated_rgba(width, height, 17);
        for pixel in opaque.pixels_mut() {
            pixel[3] = 255;
        }
        let rgb: Vec<u8> = opaque
            .as_raw()
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[r, g, b, _]| [r, g, b])
            .collect();
        let expected = resize_bicubic_pil_rgb(&rgb, width, height, 5, 4).unwrap();
        let out = resize_bicubic_pil_rgba(&DynamicImage::ImageRgba8(opaque), 5, 4);
        for (got, want) in out
            .as_raw()
            .as_chunks::<4>()
            .0
            .iter()
            .zip(expected.as_raw().as_chunks::<3>().0)
        {
            assert_eq!(got[..3], want[..], "the colour bytes of an opaque pixel");
            assert_eq!(got[3], 255, "the alpha of an opaque pixel");
        }

        let translucent = generated_rgba(width, height, 23);
        let same = resize_bicubic_pil_rgba(
            &DynamicImage::ImageRgba8(translucent.clone()),
            width,
            height,
        );
        assert_eq!(
            same.as_raw(),
            translucent.as_raw(),
            "an unchanged size is a copy of the input"
        );
    }
}
