//! The torchvision `uint8` antialiased resample over interleaved RGB bytes on
//! the targets without the Pillow-exact vector passes (x86-64 routes the torch
//! taps through those passes in [`transforms`](super::transforms)): the
//! separable kernel of `transforms` with its `TorchUint8` taps (16-bit, at the
//! finest scale that keeps the largest tap below 2^15), accumulated in 32
//! bits, on AArch64 NEON and in plain loops elsewhere. Every path computes the
//! same integer arithmetic, so the bytes are identical.
//!
//! Each output sample is `clip8((half + Σ pixel · tap) >> precision)`. A tap
//! is below 2^15 in magnitude, a product below 255 · 2^15, and the taps of
//! one output sum to one at `precision` bits (at most 22, with the negative
//! lobes of the bicubic kernel adding under a third again), so a sum stays
//! well inside an `i32`: the design of Pillow's own `INT32` resampler, which
//! the torch kernel inherits. The earlier loops accumulated in `i64` and
//! produced the same values; the tests keep that computation as the oracle.
//!
//! The AArch64 path is the module's only use of `core::arch` vector loads and
//! stores, hence the local override of the workspace-wide
//! `unsafe_code = "deny"`; every pointer it forms is bounded by an exact-size
//! chunk of a slice or by a scratch row padded for the widest read.
#![cfg_attr(target_arch = "aarch64", allow(unsafe_code))]

use super::{
    execution,
    transforms::{FixedKernel, KernelQuantization, PilResizeFilter},
};

/// One resample axis in the layout the passes read: per output index the
/// source start and tap count, the taps as `i16` at a fixed stride (a
/// multiple of four, zero-padded), on a vector architecture the same taps in
/// the layout its width pass reads ([`wide_taps`]), and the shift that undoes
/// their scale.
struct Axis {
    bounds: Vec<(usize, usize)>,
    stride: usize,
    taps: Vec<i16>,
    /// The taps of the width axis in the vector width pass's layout (sixteen
    /// lanes per group of four taps), read by that pass only; empty for the
    /// height axis and for a width axis the scalar loops read.
    #[cfg(target_arch = "aarch64")]
    taps_wide: Vec<i16>,
    precision: u32,
}

/// Which pass an [`Axis`] serves: the width pass reads the taps four-wide on
/// the vector unit, the height pass does not.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Width,
    Height,
}

/// Which rows the passes run: the host's vector rows where detected, else the
/// scalar loops. The tests run the scalar loops on purpose as well, so that an
/// architecture whose runners all have the vector unit still checks its
/// fallback.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rows {
    Detected,
    #[cfg_attr(not(test), expect(dead_code, reason = "the oracle test's second arm"))]
    Scalar,
}

/// On AArch64 only: every caller sits behind the same `cfg`, and a build for
/// another target has no vector rows to ask about.
#[cfg(target_arch = "aarch64")]
impl Rows {
    /// Whether the passes run the vector rows: the detected rows on a host
    /// whose unit is present. The width axis builds its wide taps and the
    /// passes pick their loops on this one answer, so a scalar run never
    /// allocates a table it does not read.
    #[inline]
    fn vector(self) -> bool {
        self == Rows::Detected && std::arch::is_aarch64_feature_detected!("neon")
    }
}

impl Axis {
    fn new(
        in_size: usize,
        out_size: usize,
        filter: PilResizeFilter,
        pass: Pass,
        rows: Rows,
    ) -> Self {
        let kernel = FixedKernel::new(in_size, out_size, filter, KernelQuantization::TorchUint8);
        let stride = kernel
            .taps
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .div_ceil(4)
            * 4;
        let mut taps = vec![0i16; out_size * stride];
        for (index, output_taps) in kernel.taps.iter().enumerate() {
            for (k, &tap) in output_taps.iter().enumerate() {
                // Below 2^15 in magnitude by the quantization's rule (the
                // finest scale that keeps the largest tap under it); a tap
                // outside `i16` would be a broken invariant, not a value to
                // clamp.
                debug_assert!(
                    i16::try_from(tap).is_ok(),
                    "torch uint8 tap {tap} exceeds i16"
                );
                taps[index * stride + k] = tap as i16;
            }
        }
        #[cfg(target_arch = "aarch64")]
        let taps_wide = match pass {
            Pass::Width if rows.vector() => taps
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .flat_map(wide_taps)
                .collect(),
            Pass::Width | Pass::Height => Vec::new(),
        };
        #[cfg(not(target_arch = "aarch64"))]
        let _ = (pass, rows);
        Self {
            bounds: kernel.bounds,
            stride,
            taps,
            #[cfg(target_arch = "aarch64")]
            taps_wide,
            precision: kernel.precision,
        }
    }

    #[inline]
    fn half(&self) -> i32 {
        1 << (self.precision - 1)
    }

    #[inline]
    fn clip8(&self, sum: i32) -> u8 {
        (sum >> self.precision).clamp(0, 255) as u8
    }
}

/// Four taps of the width axis as the NEON width pass reads them: each tap
/// four-wide, one lane per channel of one source pixel.
#[cfg(target_arch = "aarch64")]
fn wide_taps(group: [i16; 4]) -> [i16; 16] {
    std::array::from_fn(|lane| group[lane / 4])
}

/// `data` (`in_w` x `in_h` interleaved RGB) resampled to `out_w` x `out_h`
/// with `filter`'s kernel in the torch `uint8` quantization: the width pass
/// first, then the height pass, each skipped on an axis that keeps its size.
pub(super) fn resize_rgb(
    data: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    filter: PilResizeFilter,
) -> Vec<u8> {
    resize_rgb_with(data, in_w, in_h, out_w, out_h, filter, Rows::Detected)
}

/// [`resize_rgb`] through the scalar loops whatever the host has: the oracle
/// test's second arm.
#[cfg(test)]
fn resize_rgb_scalar(
    data: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    filter: PilResizeFilter,
) -> Vec<u8> {
    resize_rgb_with(data, in_w, in_h, out_w, out_h, filter, Rows::Scalar)
}

/// [`resize_rgb`] on the given rows.
fn resize_rgb_with(
    data: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    filter: PilResizeFilter,
    path: Rows,
) -> Vec<u8> {
    debug_assert_eq!(data.len(), in_w * in_h * 3);
    let width_axis = || Axis::new(in_w, out_w, filter, Pass::Width, path);
    let height_axis = || Axis::new(in_h, out_h, filter, Pass::Height, path);
    match (in_w != out_w, in_h != out_h) {
        // A zero-sized target has no pixels (the passes would band a zero
        // chunk); the fallible entry point refuses it, the infallible one
        // keeps its empty answer.
        _ if out_w == 0 || out_h == 0 => Vec::new(),
        (false, false) => data.to_vec(),
        (true, false) => horizontal_pass(data, in_w, in_h, out_w, &width_axis(), path),
        (false, true) => vertical_pass(data, out_w, out_h, &height_axis(), path),
        (true, true) => {
            let mid = horizontal_pass(data, in_w, in_h, out_w, &width_axis(), path);
            vertical_pass(&mid, out_w, out_h, &height_axis(), path)
        }
    }
}

/// Every input row resampled along its width; rows banded across the pool
/// (each output row depends on its input row alone, so the banding changes
/// nothing).
fn horizontal_pass(
    src: &[u8],
    in_w: usize,
    rows: usize,
    out_w: usize,
    axis: &Axis,
    path: Rows,
) -> Vec<u8> {
    let row_out = out_w * 3;
    let mut out = vec![0u8; rows * row_out];
    let tasks = execution::task_count(out.len(), rows, 32);
    if tasks <= 1 {
        horizontal_rows(src, in_w, out_w, axis, 0, &mut out, path);
        return out;
    }
    let band_rows = rows.div_ceil(tasks);
    execution::scope(|spawner| {
        let mut rest = out.as_mut_slice();
        let mut y0 = 0;
        while y0 < rows {
            let n = band_rows.min(rows - y0);
            let (band, tail) = rest.split_at_mut(n * row_out);
            rest = tail;
            let start = y0;
            spawner.spawn(move |_| horizontal_rows(src, in_w, out_w, axis, start, band, path));
            y0 += n;
        }
    });
    out
}

/// Output rows `[y0, y0 + band.len() / (out_w * 3))` of the width pass.
fn horizontal_rows(
    src: &[u8],
    in_w: usize,
    out_w: usize,
    axis: &Axis,
    y0: usize,
    band: &mut [u8],
    path: Rows,
) {
    let row_in = in_w * 3;
    #[cfg(not(target_arch = "aarch64"))]
    let _ = path;
    #[cfg(target_arch = "aarch64")]
    if path.vector() {
        // Scratch for the input row as four bytes per pixel, padded by the
        // widest read of a tap group (three pixels past the row's end).
        let mut rgbx = vec![0u8; (in_w + 4) * 4];
        for (i, out_row) in band.chunks_exact_mut(out_w * 3).enumerate() {
            let row = &src[(y0 + i) * row_in..(y0 + i + 1) * row_in];
            // SAFETY: `vector` holds only when NEON was detected.
            unsafe { neon::horizontal_row(row, &mut rgbx, axis, out_row) };
        }
        return;
    }
    for (i, out_row) in band.chunks_exact_mut(out_w * 3).enumerate() {
        let row = &src[(y0 + i) * row_in..(y0 + i + 1) * row_in];
        horizontal_row_scalar(row, axis, out_row);
    }
}

fn horizontal_row_scalar(row: &[u8], axis: &Axis, out_row: &mut [u8]) {
    let half = axis.half();
    for (index, out) in out_row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let (x0, n) = axis.bounds[index];
        let taps = &axis.taps[index * axis.stride..index * axis.stride + n];
        let (mut r, mut g, mut b) = (half, half, half);
        for (pixel, &tap) in row[x0 * 3..(x0 + n) * 3]
            .as_chunks::<3>()
            .0
            .iter()
            .zip(taps)
        {
            let tap = i32::from(tap);
            r += i32::from(pixel[0]) * tap;
            g += i32::from(pixel[1]) * tap;
            b += i32::from(pixel[2]) * tap;
        }
        out[0] = axis.clip8(r);
        out[1] = axis.clip8(g);
        out[2] = axis.clip8(b);
    }
}

/// Every output row resampled along the height; rows banded across the pool.
fn vertical_pass(src: &[u8], width: usize, out_h: usize, axis: &Axis, path: Rows) -> Vec<u8> {
    let row_len = width * 3;
    let mut out = vec![0u8; out_h * row_len];
    let tasks = execution::task_count(out.len(), out_h, 32);
    if tasks <= 1 {
        vertical_rows(src, row_len, axis, 0, &mut out, path);
        return out;
    }
    let band_rows = out_h.div_ceil(tasks);
    execution::scope(|spawner| {
        let mut rest = out.as_mut_slice();
        let mut y0 = 0;
        while y0 < out_h {
            let n = band_rows.min(out_h - y0);
            let (band, tail) = rest.split_at_mut(n * row_len);
            rest = tail;
            let start = y0;
            spawner.spawn(move |_| vertical_rows(src, row_len, axis, start, band, path));
            y0 += n;
        }
    });
    out
}

/// Output rows `[y0, y0 + band.len() / row_len)` of the height pass.
fn vertical_rows(src: &[u8], row_len: usize, axis: &Axis, y0: usize, band: &mut [u8], path: Rows) {
    #[cfg(not(target_arch = "aarch64"))]
    let _ = path;
    #[cfg(target_arch = "aarch64")]
    if path.vector() {
        for (i, out_row) in band.chunks_exact_mut(row_len).enumerate() {
            let (start, n) = axis.bounds[y0 + i];
            let taps = &axis.taps[(y0 + i) * axis.stride..(y0 + i) * axis.stride + n];
            let rows = &src[start * row_len..(start + n) * row_len];
            // SAFETY: `vector` holds only when NEON was detected.
            unsafe { neon::vertical_row(rows, taps, axis, out_row) };
        }
        return;
    }
    let mut sums = vec![0i32; row_len];
    for (i, out_row) in band.chunks_exact_mut(row_len).enumerate() {
        let (start, n) = axis.bounds[y0 + i];
        let taps = &axis.taps[(y0 + i) * axis.stride..(y0 + i) * axis.stride + n];
        sums.fill(axis.half());
        for (k, &tap) in taps.iter().enumerate() {
            let tap = i32::from(tap);
            let row = &src[(start + k) * row_len..(start + k + 1) * row_len];
            for (sum, &value) in sums.iter_mut().zip(row) {
                *sum += i32::from(value) * tap;
            }
        }
        for (out, &sum) in out_row.iter_mut().zip(&sums) {
            *out = axis.clip8(sum);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        int16x8_t, int32x4_t, uint16x4_t, uint8x16x4_t, vcombine_u16, vcombine_u8, vdupq_n_s16,
        vdupq_n_s32, vdupq_n_u8, vget_high_s16, vget_lane_u32, vget_low_s16, vget_low_u8,
        vld1q_s16, vld1q_u8, vld3q_u8, vmlal_high_s16, vmlal_s16, vmovl_high_u8, vmovl_u8,
        vqmovn_u16, vqmovun_s32, vreinterpret_u32_u8, vreinterpretq_s16_u16, vshlq_s32, vst1q_u8,
        vst4q_u8,
    };

    use super::Axis;

    /// `(sum >> precision)` clamped to a byte, for four sums.
    #[inline]
    #[target_feature(enable = "neon")]
    fn clip8x4(sums: int32x4_t, shift: int32x4_t) -> uint16x4_t {
        vqmovun_s32(vshlq_s32(sums, shift))
    }

    /// One row of the width pass: the row is spread to four bytes per pixel
    /// in `rgbx` (zero padded), then every output pixel sums its taps four
    /// source pixels at a time, the taps four-wide.
    #[target_feature(enable = "neon")]
    pub(super) fn horizontal_row(row: &[u8], rgbx: &mut [u8], axis: &Axis, out_row: &mut [u8]) {
        let pixels = row.len() / 3;
        debug_assert!(rgbx.len() >= (pixels + 4) * 4);
        let (chunks, rest) = row.as_chunks::<48>();
        // The scratch splits where the row's whole chunks end, so the tail
        // pixels follow the vector-written ones; the padding stays zero.
        let (wide, wide_rest) = rgbx.split_at_mut(chunks.len() * 64);
        let zero = vdupq_n_u8(0);
        for (chunk, wide) in chunks.iter().zip(wide.as_chunks_mut::<64>().0.iter_mut()) {
            // SAFETY: `chunk` is 48 bytes (16 pixels), `wide` 64 bytes.
            unsafe {
                let planes = vld3q_u8(chunk.as_ptr());
                vst4q_u8(
                    wide.as_mut_ptr(),
                    uint8x16x4_t(planes.0, planes.1, planes.2, zero),
                );
            }
        }
        for (pixel, wide) in rest
            .as_chunks::<3>()
            .0
            .iter()
            .zip(wide_rest.as_chunks_mut::<4>().0.iter_mut())
        {
            wide[..3].copy_from_slice(pixel);
            wide[3] = 0;
        }
        let out_w = out_row.len() / 3;
        let half = vdupq_n_s32(axis.half());
        let shift = vdupq_n_s32(-(axis.precision as i32));
        for index in 0..out_w {
            let (x0, n) = axis.bounds[index];
            let groups = n.div_ceil(4);
            let taps =
                &axis.taps_wide[index * axis.stride * 4..(index * axis.stride + groups * 4) * 4];
            let source = &rgbx[x0 * 4..(x0 + groups * 4) * 4];
            let mut sums = half;
            for (pixels, taps) in source
                .as_chunks::<16>()
                .0
                .iter()
                .zip(taps.as_chunks::<16>().0)
            {
                // SAFETY: `pixels` is 16 bytes (four pixels), `taps` 16
                // `i16`s (four taps, each four-wide).
                let (pixels, first, second) = unsafe {
                    (
                        vld1q_u8(pixels.as_ptr()),
                        vld1q_s16(taps.as_ptr()),
                        vld1q_s16(taps.as_ptr().add(8)),
                    )
                };
                let low: int16x8_t = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pixels)));
                let high: int16x8_t = vreinterpretq_s16_u16(vmovl_high_u8(pixels));
                sums = vmlal_s16(sums, vget_low_s16(low), vget_low_s16(first));
                sums = vmlal_high_s16(sums, low, first);
                sums = vmlal_s16(sums, vget_low_s16(high), vget_low_s16(second));
                sums = vmlal_high_s16(sums, high, second);
            }
            let narrow = clip8x4(sums, shift);
            let bytes = vqmovn_u16(vcombine_u16(narrow, narrow));
            // The pixel's three bytes (and a fourth, the clipped value of the
            // first lane again) as a word, written through the slice: a
            // 3-byte stride is 4-byte aligned for one pixel in four only, so
            // no typed `u32` store.
            let word = vget_lane_u32::<0>(vreinterpret_u32_u8(bytes)).to_ne_bytes();
            if index + 1 < out_w {
                // The fourth byte is the next pixel's first, written again by
                // the next iteration.
                out_row[index * 3..index * 3 + 4].copy_from_slice(&word);
            } else {
                out_row[index * 3..index * 3 + 3].copy_from_slice(&word[..3]);
            }
        }
    }

    /// One row of the height pass: sixteen output bytes at a time across the
    /// taps, four 32-bit accumulators each; the row's last bytes in scalar.
    #[target_feature(enable = "neon")]
    pub(super) fn vertical_row(rows: &[u8], taps: &[i16], axis: &Axis, out_row: &mut [u8]) {
        let row_len = out_row.len();
        let half = vdupq_n_s32(axis.half());
        let shift = vdupq_n_s32(-(axis.precision as i32));
        let (chunks, rest) = out_row.as_chunks_mut::<16>();
        for (x, out) in chunks.iter_mut().enumerate() {
            let (mut s0, mut s1, mut s2, mut s3) = (half, half, half, half);
            for (k, &tap) in taps.iter().enumerate() {
                let tap = vdupq_n_s16(tap);
                let offset = k * row_len + x * 16;
                // SAFETY: `rows` holds `taps.len()` rows of `row_len` bytes
                // and `offset + 16 <= (k + 1) * row_len`.
                let pixels = unsafe { vld1q_u8(rows.as_ptr().add(offset)) };
                let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pixels)));
                let high = vreinterpretq_s16_u16(vmovl_high_u8(pixels));
                s0 = vmlal_s16(s0, vget_low_s16(low), vget_low_s16(tap));
                s1 = vmlal_s16(s1, vget_high_s16(low), vget_low_s16(tap));
                s2 = vmlal_s16(s2, vget_low_s16(high), vget_low_s16(tap));
                s3 = vmlal_s16(s3, vget_high_s16(high), vget_low_s16(tap));
            }
            let bytes = vcombine_u8(
                vqmovn_u16(vcombine_u16(clip8x4(s0, shift), clip8x4(s1, shift))),
                vqmovn_u16(vcombine_u16(clip8x4(s2, shift), clip8x4(s3, shift))),
            );
            // SAFETY: `out` is exactly 16 bytes.
            unsafe { vst1q_u8(out.as_mut_ptr(), bytes) };
        }
        let done = chunks.len() * 16;
        for (i, out) in rest.iter_mut().enumerate() {
            let x = done + i;
            let mut sum = axis.half();
            for (k, &tap) in taps.iter().enumerate() {
                sum += i32::from(rows[k * row_len + x]) * i32::from(tap);
            }
            *out = axis.clip8(sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        test_support::{assert_same_bytes, gradient_rgb},
        vision::transforms::per_sample_reference_channels,
    };

    /// Up- and downscales, extreme aspect ratios, identity axes, widths that
    /// leave vector tails (rows not a multiple of 16 or 32 bytes, pixel
    /// counts not a multiple of 4) and a full-size image: every byte equals
    /// the `i64` per-sample computation on every target, through the rows
    /// this host detects and through the scalar loops.
    #[test]
    fn every_byte_equals_the_per_sample_reference() {
        let cases = [
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
            (1296, 972, 1280, 960),
        ];
        for (seed, &(in_w, in_h, out_w, out_h)) in cases.iter().enumerate() {
            let data = gradient_rgb(in_w, in_h, seed * 13);
            let expected = per_sample_reference_channels::<3>(
                &data,
                in_w,
                in_h,
                out_w,
                out_h,
                PilResizeFilter::Bicubic,
                KernelQuantization::TorchUint8,
            );
            let detected = resize_rgb(&data, in_w, in_h, out_w, out_h, PilResizeFilter::Bicubic);
            assert_same_bytes(
                &detected,
                &expected,
                &format!("{in_w}x{in_h} -> {out_w}x{out_h}, the detected rows"),
            );
            let scalar =
                resize_rgb_scalar(&data, in_w, in_h, out_w, out_h, PilResizeFilter::Bicubic);
            assert_same_bytes(
                &scalar,
                &expected,
                &format!("{in_w}x{in_h} -> {out_w}x{out_h}, the scalar rows"),
            );
        }
    }

    /// A zero-sized target is an empty buffer, as the previous resampler
    /// answered (the infallible entry point keeps that contract).
    #[test]
    fn a_zero_sized_target_is_empty() {
        let data = gradient_rgb(9, 4, 2);
        assert!(resize_rgb(&data, 9, 4, 0, 4, PilResizeFilter::Bicubic).is_empty());
        assert!(resize_rgb(&data, 9, 4, 9, 0, PilResizeFilter::Bicubic).is_empty());
        assert!(resize_rgb(&data, 9, 4, 0, 0, PilResizeFilter::Bicubic).is_empty());
    }

    /// The identity (no change on either axis) is a plain copy.
    #[test]
    fn an_unchanged_size_copies_the_bytes() {
        let data = gradient_rgb(9, 4, 1);
        assert_eq!(
            resize_rgb(&data, 9, 4, 9, 4, PilResizeFilter::Bicubic),
            data
        );
    }

    /// The wide taps serve the vector width pass alone: a width axis the
    /// scalar loops read carries none and allocates none, so a host without
    /// the vector unit (or the scalar arm above) builds no table it never
    /// reads. A height axis never has them; the vector width axis holds four
    /// lanes per tap.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn a_width_axis_the_scalar_loops_read_carries_no_wide_taps() {
        let axis = |pass, rows| Axis::new(640, 320, PilResizeFilter::Bicubic, pass, rows);
        let scalar = axis(Pass::Width, Rows::Scalar);
        assert!(
            scalar.taps_wide.is_empty(),
            "wide taps built for the scalar loops: {} lanes",
            scalar.taps_wide.len()
        );
        assert_eq!(
            scalar.taps_wide.capacity(),
            0,
            "a wide-tap buffer allocated for the scalar loops"
        );
        let height = axis(Pass::Height, Rows::Detected);
        assert!(
            height.taps_wide.is_empty(),
            "wide taps built for a height axis"
        );
        if Rows::Detected.vector() {
            let vector = axis(Pass::Width, Rows::Detected);
            assert_eq!(
                vector.taps_wide.len(),
                vector.taps.len() * 4,
                "the vector width pass reads four lanes per tap"
            );
        }
    }
}
