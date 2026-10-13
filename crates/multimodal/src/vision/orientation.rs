//! Applying an image's orientation tag (the EXIF orientation the decoders
//! report) to decoded RGB8 pixels: the eight transforms with the geometry of
//! `DynamicImage::apply_orientation`, as tile transposes and row reversals on
//! the vector unit (AArch64) and plain loops elsewhere. Pure permutations of
//! the pixels, so every path writes the same bytes; images of other types
//! take the `image` crate's own transform.
//!
//! The AArch64 path is the module's only use of `core::arch` vector loads and
//! stores, hence the local override of the workspace-wide
//! `unsafe_code = "deny"`; every pointer it forms comes from an exact-size
//! slice (a 48-byte run), so a wrong length or offset is a bounds check,
//! never an out-of-bounds vector access.
#![cfg_attr(target_arch = "aarch64", allow(unsafe_code))]

use image::{metadata::Orientation, DynamicImage, RgbImage};

/// `image` with `orientation` applied: the pixels `DynamicImage::apply_orientation`
/// produces, by the fast paths for RGB8 and the `image` crate for the rest.
pub(crate) fn oriented(image: DynamicImage, orientation: Orientation) -> DynamicImage {
    if orientation == Orientation::NoTransforms {
        return image;
    }
    match image {
        DynamicImage::ImageRgb8(rgb) => DynamicImage::ImageRgb8(oriented_rgb8(&rgb, orientation)),
        mut other => {
            other.apply_orientation(orientation);
            other
        }
    }
}

/// Where a transposed tile's rows and columns land: the four 90-degree
/// orientations are the transpose with its rows, its columns, or both in
/// reverse order.
#[derive(Clone, Copy)]
struct Transpose {
    reverse_rows: bool,
    reverse_columns: bool,
}

fn oriented_rgb8(image: &RgbImage, orientation: Orientation) -> RgbImage {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let src = image.as_raw();
    let (out_w, out_h, out) = match orientation {
        Orientation::NoTransforms => (w, h, src.clone()),
        Orientation::FlipHorizontal => (w, h, flipped(src, w, h, false, true)),
        Orientation::FlipVertical => (w, h, flipped(src, w, h, true, false)),
        Orientation::Rotate180 => (w, h, flipped(src, w, h, true, true)),
        // out[x][y] = in[y][x]
        Orientation::Rotate90FlipH => (
            h,
            w,
            transposed(
                src,
                w,
                h,
                Transpose {
                    reverse_rows: false,
                    reverse_columns: false,
                },
            ),
        ),
        // clockwise: out[x][h - 1 - y] = in[y][x]
        Orientation::Rotate90 => (
            h,
            w,
            transposed(
                src,
                w,
                h,
                Transpose {
                    reverse_rows: false,
                    reverse_columns: true,
                },
            ),
        ),
        // counter-clockwise: out[w - 1 - x][y] = in[y][x]
        Orientation::Rotate270 => (
            h,
            w,
            transposed(
                src,
                w,
                h,
                Transpose {
                    reverse_rows: true,
                    reverse_columns: false,
                },
            ),
        ),
        // out[w - 1 - x][h - 1 - y] = in[y][x]
        Orientation::Rotate270FlipH => (
            h,
            w,
            transposed(
                src,
                w,
                h,
                Transpose {
                    reverse_rows: true,
                    reverse_columns: true,
                },
            ),
        ),
    };
    #[expect(
        clippy::expect_used,
        reason = "the buffer is exactly out_w * out_h * 3 bytes by construction"
    )]
    RgbImage::from_raw(out_w as u32, out_h as u32, out).expect("oriented image buffer size")
}

/// Rows in reverse order (`rows`) and/or the pixels of each row in reverse
/// order (`pixels`).
fn flipped(src: &[u8], w: usize, h: usize, rows: bool, pixels: bool) -> Vec<u8> {
    let row_len = w * 3;
    let mut out = vec![0u8; src.len()];
    for (y, row) in src.chunks_exact(row_len).enumerate() {
        let target = if rows { h - 1 - y } else { y };
        let out_row = &mut out[target * row_len..(target + 1) * row_len];
        if pixels {
            reverse_pixels(row, out_row);
        } else {
            out_row.copy_from_slice(row);
        }
    }
    out
}

/// `out_row` = the pixels of `row` in reverse order.
fn reverse_pixels(row: &[u8], out_row: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: NEON was detected.
        unsafe { neon::reverse_pixels(row, out_row) };
        return;
    }
    reverse_pixels_scalar(row, out_row);
}

fn reverse_pixels_scalar(row: &[u8], out_row: &mut [u8]) {
    for (pixel, out) in row
        .as_chunks::<3>()
        .0
        .iter()
        .zip(out_row.as_chunks_mut::<3>().0.iter_mut().rev())
    {
        *out = *pixel;
    }
}

/// The transpose of `src` (`w` x `h`) into an `h` x `w` buffer, rows and
/// columns of the result reversed as `layout` says: 16 x 16 pixel tiles on
/// the vector unit, the border tiles pixel by pixel.
fn transposed(src: &[u8], w: usize, h: usize, layout: Transpose) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    let (out_w, out_h) = (h, w);
    // Destination row of source column x, destination column of source row y.
    let row_of = |x: usize| {
        if layout.reverse_rows {
            out_h - 1 - x
        } else {
            x
        }
    };
    let col_of = |y: usize| {
        if layout.reverse_columns {
            out_w - 1 - y
        } else {
            y
        }
    };
    let (tiles_x, tiles_y) = (w / 16, h / 16);
    #[cfg(target_arch = "aarch64")]
    let vector = std::arch::is_aarch64_feature_detected!("neon");
    #[cfg(not(target_arch = "aarch64"))]
    let vector = false;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            if vector {
                #[cfg(target_arch = "aarch64")]
                // SAFETY: NEON was detected (the tile's runs are bounds-checked
                // slices inside).
                unsafe {
                    neon::transpose_tile(src, w, &mut out, out_w, tx * 16, ty * 16, layout);
                }
            } else {
                transpose_block(
                    src,
                    w,
                    &mut out,
                    out_w,
                    tx * 16,
                    ty * 16,
                    16,
                    16,
                    &row_of,
                    &col_of,
                );
            }
        }
    }
    // The border: the columns past the last whole tile for every row, and the
    // rows past the last whole tile for the columns the tiles covered.
    if tiles_x * 16 < w {
        transpose_block(
            src,
            w,
            &mut out,
            out_w,
            tiles_x * 16,
            0,
            w - tiles_x * 16,
            h,
            &row_of,
            &col_of,
        );
    }
    if tiles_y * 16 < h {
        transpose_block(
            src,
            w,
            &mut out,
            out_w,
            0,
            tiles_y * 16,
            tiles_x * 16,
            h - tiles_y * 16,
            &row_of,
            &col_of,
        );
    }
    out
}

/// Pixel-by-pixel transpose of the `cols` x `rows` block at (`x0`, `y0`).
#[expect(
    clippy::too_many_arguments,
    reason = "a block of a transpose: two buffers, origin, size, the mappings"
)]
fn transpose_block(
    src: &[u8],
    w: usize,
    out: &mut [u8],
    out_w: usize,
    x0: usize,
    y0: usize,
    cols: usize,
    rows: usize,
    row_of: &dyn Fn(usize) -> usize,
    col_of: &dyn Fn(usize) -> usize,
) {
    for y in y0..y0 + rows {
        let col = col_of(y);
        let row = &src[(y * w + x0) * 3..(y * w + x0 + cols) * 3];
        for (i, pixel) in row.as_chunks::<3>().0.iter().enumerate() {
            let o = (row_of(x0 + i) * out_w + col) * 3;
            out[o..o + 3].copy_from_slice(pixel);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        uint8x16_t, uint8x16x3_t, vdupq_n_u8, vextq_u8, vld3q_u8, vreinterpretq_u16_u8,
        vreinterpretq_u32_u8, vreinterpretq_u64_u8, vreinterpretq_u8_u16, vreinterpretq_u8_u32,
        vreinterpretq_u8_u64, vrev64q_u8, vst3q_u8, vtrn1q_u16, vtrn1q_u32, vtrn1q_u64, vtrn1q_u8,
        vtrn2q_u16, vtrn2q_u32, vtrn2q_u64, vtrn2q_u8,
    };

    use super::Transpose;

    /// The pixels of `row` in reverse order into `out_row` (cut to `row`'s
    /// length): sixteen pixels per de-interleaving load, each plane reversed,
    /// stored into the mirrored 48-byte chunk of the destination; the
    /// remainder (the source's tail, the destination's head) pixel by pixel.
    /// Every pointer comes from a 48-byte chunk of a slice, so a short
    /// `out_row` is a bounds-check panic, not an out-of-bounds store.
    #[target_feature(enable = "neon")]
    pub(super) fn reverse_pixels(row: &[u8], out_row: &mut [u8]) {
        let out_row = &mut out_row[..row.len()];
        let (chunks, rest) = row.as_chunks::<48>();
        let (out_rest, out_chunks) = out_row.as_rchunks_mut::<48>();
        for (chunk, out) in chunks.iter().zip(out_chunks.iter_mut().rev()) {
            // SAFETY: both pointers come from 48-byte arrays.
            unsafe {
                let planes = vld3q_u8(chunk.as_ptr());
                vst3q_u8(
                    out.as_mut_ptr(),
                    uint8x16x3_t(reversed(planes.0), reversed(planes.1), reversed(planes.2)),
                );
            }
        }
        super::reverse_pixels_scalar(rest, out_rest);
    }

    /// The sixteen bytes of a vector in reverse order.
    #[inline]
    #[target_feature(enable = "neon")]
    fn reversed(v: uint8x16_t) -> uint8x16_t {
        let halves = vrev64q_u8(v);
        vextq_u8::<8>(halves, halves)
    }

    /// A 16 x 16 byte matrix held as sixteen row vectors, transposed: four
    /// rounds of pairwise interleaves at 8, 16, 32 and 64 bits, after which
    /// vector `c` holds column `c`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn transpose16(mut t: [uint8x16_t; 16]) -> [uint8x16_t; 16] {
        for i in (0..16).step_by(2) {
            let (a, b) = (t[i], t[i + 1]);
            t[i] = vtrn1q_u8(a, b);
            t[i + 1] = vtrn2q_u8(a, b);
        }
        for i in [0, 1, 4, 5, 8, 9, 12, 13] {
            let (a, b) = (vreinterpretq_u16_u8(t[i]), vreinterpretq_u16_u8(t[i + 2]));
            t[i] = vreinterpretq_u8_u16(vtrn1q_u16(a, b));
            t[i + 2] = vreinterpretq_u8_u16(vtrn2q_u16(a, b));
        }
        for i in [0, 1, 2, 3, 8, 9, 10, 11] {
            let (a, b) = (vreinterpretq_u32_u8(t[i]), vreinterpretq_u32_u8(t[i + 4]));
            t[i] = vreinterpretq_u8_u32(vtrn1q_u32(a, b));
            t[i + 4] = vreinterpretq_u8_u32(vtrn2q_u32(a, b));
        }
        for i in 0..8 {
            let (a, b) = (vreinterpretq_u64_u8(t[i]), vreinterpretq_u64_u8(t[i + 8]));
            t[i] = vreinterpretq_u8_u64(vtrn1q_u64(a, b));
            t[i + 8] = vreinterpretq_u8_u64(vtrn2q_u64(a, b));
        }
        t
    }

    /// The 16 x 16 pixel tile at (`x0`, `y0`) of `src` (`w` pixels wide)
    /// transposed into `out` (`out_w` pixels wide) under `layout`: the
    /// source rows are loaded in the order the destination columns take, so
    /// the transposed vectors store straight into their destination rows.
    #[target_feature(enable = "neon")]
    pub(super) fn transpose_tile(
        src: &[u8],
        w: usize,
        out: &mut [u8],
        out_w: usize,
        x0: usize,
        y0: usize,
        layout: Transpose,
    ) {
        let h = src.len() / (w * 3);
        let out_h = w;
        let zero = vdupq_n_u8(0);
        let (mut r, mut g, mut b) = ([zero; 16], [zero; 16], [zero; 16]);
        for k in 0..16 {
            // Destination column `col0 + k` holds source row `y`.
            let y = if layout.reverse_columns {
                y0 + 15 - k
            } else {
                y0 + k
            };
            // The source run: a 48-byte slice (a wrong origin is a bounds
            // check, not an out-of-bounds load).
            let start = (y * w + x0) * 3;
            let run = &src[start..start + 48];
            // SAFETY: `run` is 48 bytes.
            let planes = unsafe { vld3q_u8(run.as_ptr()) };
            r[k] = planes.0;
            g[k] = planes.1;
            b[k] = planes.2;
        }
        let (r, g, b) = (transpose16(r), transpose16(g), transpose16(b));
        let col0 = if layout.reverse_columns {
            h - 16 - y0
        } else {
            y0
        };
        for j in 0..16 {
            let x = x0 + j;
            let row = if layout.reverse_rows {
                out_h - 1 - x
            } else {
                x
            };
            // The destination run, a 48-byte slice as above.
            let start = (row * out_w + col0) * 3;
            let run = &mut out[start..start + 48];
            // SAFETY: `run` is 48 bytes.
            unsafe {
                vst3q_u8(run.as_mut_ptr(), uint8x16x3_t(r[j], g[j], b[j]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    fn gradient(w: u32, h: u32, seed: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            Rgb([
                ((seed + x * 7 + y * 3) % 256) as u8,
                ((seed + x * 5 + y * 11) % 256) as u8,
                ((seed + x + y * 2) % 256) as u8,
            ])
        })
    }

    const ORIENTATIONS: [Orientation; 8] = [
        Orientation::NoTransforms,
        Orientation::FlipHorizontal,
        Orientation::FlipVertical,
        Orientation::Rotate180,
        Orientation::Rotate90,
        Orientation::Rotate270,
        Orientation::Rotate90FlipH,
        Orientation::Rotate270FlipH,
    ];

    /// Every orientation on sizes with whole tiles, tails on either axis and
    /// a single pixel: the same pixels `apply_orientation` produces.
    #[test]
    fn every_orientation_matches_the_image_crates_transform() {
        for (seed, &(w, h)) in [
            (1, 1),
            (16, 16),
            (17, 16),
            (16, 17),
            (37, 23),
            (33, 65),
            (64, 48),
            (100, 20),
        ]
        .iter()
        .enumerate()
        {
            let image = gradient(w, h, seed as u32 * 17);
            for orientation in ORIENTATIONS {
                let expected = {
                    let mut copy = DynamicImage::ImageRgb8(image.clone());
                    copy.apply_orientation(orientation);
                    copy
                };
                let actual = oriented(DynamicImage::ImageRgb8(image.clone()), orientation);
                assert_eq!(
                    (actual.width(), actual.height()),
                    (expected.width(), expected.height()),
                    "{w}x{h} {orientation:?}"
                );
                assert!(
                    actual.as_bytes() == expected.as_bytes(),
                    "{w}x{h} {orientation:?}: pixels differ"
                );
            }
        }
    }

    /// Other image types go through the `image` crate's transform unchanged.
    #[test]
    fn other_image_types_take_the_crates_transform() {
        let rgba = image::RgbaImage::from_fn(9, 5, |x, y| image::Rgba([x as u8, y as u8, 7, 200]));
        let mut expected = DynamicImage::ImageRgba8(rgba.clone());
        expected.apply_orientation(Orientation::Rotate90);
        let actual = oriented(DynamicImage::ImageRgba8(rgba), Orientation::Rotate90);
        assert!(actual.as_bytes() == expected.as_bytes());
    }
}
