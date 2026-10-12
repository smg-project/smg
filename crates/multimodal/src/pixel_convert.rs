//! Pixel bytes to wire words: every byte of a channel maps to the bits of its
//! wire value, `PixelNorm::table_f32` (the reference processors' per-element
//! arithmetic) rounded to the wire's float width. The tables are the single
//! source of the values. This module moves them: a scalar table lookup per
//! channel block on every target, and on AArch64 a vector formula that is
//! used for a write only after it has reproduced that write's tables for all
//! 256 bytes of all three channels, so the bytes written are the tables'
//! whichever path runs.
//!
//! Why not a vector table lookup: a 256-entry table is 512 bytes per output
//! byte plane, and the vector unit's byte permutes index 64 bytes at a time,
//! which leaves such a lookup slower than the scalar loop; the arithmetic is
//! a handful of lane-wise operations per 16 pixels.
//!
//! The AArch64 path is the crate's only use of `core::arch` vector loads and
//! stores, hence the local override of the workspace-wide
//! `unsafe_code = "deny"`; every pointer it forms comes from an exact-size
//! chunk of a slice.
#![cfg_attr(target_arch = "aarch64", allow(unsafe_code))]

use crate::encoder_inputs::PixelNorm;

/// Writes `tables[channel][byte]` as a little-endian `u16` for every byte of
/// `src` (`dst` holds two bytes per source byte). Consecutive runs of `block`
/// source bytes belong to one channel, the channels cycling 0, 1, 2 from the
/// start of `src` (the layout of a `[C, temporal, patch_h, patch_w]` patch
/// row, `block` being `PixelNorm::channel_block`).
pub(crate) fn write_u16(dst: &mut [u8], src: &[u8], block: usize, tables: &[[u16; 256]; 3]) {
    debug_assert_eq!(dst.len(), src.len() * 2);
    for ((dst, src), table) in channel_blocks(dst, src, block, 2).zip(tables.iter().cycle()) {
        for (d, &v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
            *d = table[v as usize].to_le_bytes();
        }
    }
}

/// [`write_u16`] for four-byte words.
pub(crate) fn write_u32(dst: &mut [u8], src: &[u8], block: usize, tables: &[[u32; 256]; 3]) {
    debug_assert_eq!(dst.len(), src.len() * 4);
    for ((dst, src), table) in channel_blocks(dst, src, block, 4).zip(tables.iter().cycle()) {
        for (d, &v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
            *d = table[v as usize].to_le_bytes();
        }
    }
}

/// [`write_u16`] for the `bfloat16` tables of `norm`: the vector formula
/// where it reproduces them, the table lookup elsewhere.
pub(crate) fn write_bf16(
    dst: &mut [u8],
    src: &[u8],
    block: usize,
    norm: &PixelNorm,
    tables: &[[u16; 256]; 3],
) {
    #[cfg(target_arch = "aarch64")]
    if let Some(formula) = neon::Bf16Formula::reproducing(norm, tables) {
        debug_assert_eq!(dst.len(), src.len() * 2);
        for ((dst, src), channel) in channel_blocks(dst, src, block, 2).zip((0..3).cycle()) {
            formula.write(dst, src, channel);
        }
        return;
    }
    let _ = norm;
    write_u16(dst, src, block, tables);
}

/// [`write_u32`] for the `f32` tables of `norm`: the vector formula where it
/// reproduces them, the table lookup elsewhere.
pub(crate) fn write_f32(
    dst: &mut [u8],
    src: &[u8],
    block: usize,
    norm: &PixelNorm,
    tables: &[[u32; 256]; 3],
) {
    #[cfg(target_arch = "aarch64")]
    if let Some(formula) = neon::F32Formula::reproducing(norm, tables) {
        debug_assert_eq!(dst.len(), src.len() * 4);
        for ((dst, src), channel) in channel_blocks(dst, src, block, 4).zip((0..3).cycle()) {
            formula.write(dst, src, channel);
        }
        return;
    }
    let _ = norm;
    write_u32(dst, src, block, tables);
}

/// `(dst, src)` per channel block: `block` source bytes and their `width`
/// bytes each of output.
fn channel_blocks<'a>(
    dst: &'a mut [u8],
    src: &'a [u8],
    block: usize,
    width: usize,
) -> impl Iterator<Item = (&'a mut [u8], &'a [u8])> {
    let block = block.max(1);
    dst.chunks_mut(block * width).zip(src.chunks(block))
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        float32x4_t, uint32x4_t, uint8x16_t, vcvtq_f32_u32, vdivq_f32, vdupq_n_f32, vget_low_u16,
        vget_low_u8, vld1q_u8, vmovl_high_u16, vmovl_high_u8, vmovl_u16, vmovl_u8, vmulq_f32,
        vreinterpretq_u32_f32, vrshrn_high_n_u32, vrshrn_n_u32, vst1q_f32, vst1q_u16, vsubq_f32,
    };

    use super::PixelNorm;

    /// The bytes 0..=255 in order: the input that produces a whole table.
    fn every_byte() -> [u8; 256] {
        std::array::from_fn(|value| value as u8)
    }

    /// `(byte * scale - offset) * (1 / divisor)` in `f32`, rounded to
    /// `bfloat16` half-up on the bit pattern: cheaper than a division and
    /// than round-to-nearest-even, and exactly the table's value whenever
    /// [`Self::reproducing`] says so, which is the only time it runs.
    pub(super) struct Bf16Formula {
        scale: f32,
        offset: [f32; 3],
        reciprocal: [f32; 3],
    }

    impl Bf16Formula {
        /// The formula for `norm`, if it reproduces `tables` for every byte of
        /// every channel (768 values, checked with the same code that writes).
        pub(super) fn reproducing(norm: &PixelNorm, tables: &[[u16; 256]; 3]) -> Option<Self> {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return None;
            }
            let formula = Self {
                scale: norm.scale,
                offset: norm.offset,
                reciprocal: norm.divisor.map(|divisor| 1.0 / divisor),
            };
            let input = every_byte();
            let mut output = [0u8; 512];
            (0..3)
                .all(|channel| {
                    formula.write(&mut output, &input, channel);
                    output
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(&tables[channel])
                        .all(|(written, &value)| *written == value.to_le_bytes())
                })
                .then_some(formula)
        }

        /// Writes `channel`'s words for `src` into `dst` (two bytes each).
        pub(super) fn write(&self, dst: &mut [u8], src: &[u8], channel: usize) {
            // SAFETY: NEON was detected in `reproducing`, the only constructor.
            unsafe {
                write_bf16(
                    dst,
                    src,
                    self.scale,
                    self.offset[channel],
                    self.reciprocal[channel],
                );
            }
        }
    }

    /// `(byte * scale - offset) / divisor` in `f32` on the vector unit.
    pub(super) struct F32Formula {
        scale: f32,
        offset: [f32; 3],
        divisor: [f32; 3],
    }

    impl F32Formula {
        /// The formula for `norm`, if it reproduces `tables` for every byte of
        /// every channel.
        pub(super) fn reproducing(norm: &PixelNorm, tables: &[[u32; 256]; 3]) -> Option<Self> {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return None;
            }
            let formula = Self {
                scale: norm.scale,
                offset: norm.offset,
                divisor: norm.divisor,
            };
            let input = every_byte();
            let mut output = [0u8; 1024];
            (0..3)
                .all(|channel| {
                    formula.write(&mut output, &input, channel);
                    output
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(&tables[channel])
                        .all(|(written, &value)| *written == value.to_le_bytes())
                })
                .then_some(formula)
        }

        /// Writes `channel`'s words for `src` into `dst` (four bytes each).
        pub(super) fn write(&self, dst: &mut [u8], src: &[u8], channel: usize) {
            // SAFETY: NEON was detected in `reproducing`, the only constructor.
            unsafe {
                write_f32(
                    dst,
                    src,
                    self.scale,
                    self.offset[channel],
                    self.divisor[channel],
                );
            }
        }
    }

    /// 16 bytes as four vectors of four `f32`s.
    #[inline]
    #[target_feature(enable = "neon")]
    fn widen(bytes: uint8x16_t) -> [float32x4_t; 4] {
        let low = vmovl_u8(vget_low_u8(bytes));
        let high = vmovl_high_u8(bytes);
        [
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(low))),
            vcvtq_f32_u32(vmovl_high_u16(low)),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(high))),
            vcvtq_f32_u32(vmovl_high_u16(high)),
        ]
    }

    /// `value * scale - offset`, the multiplication skipped for a scale of 1
    /// (where it is exact anyway).
    #[inline]
    #[target_feature(enable = "neon")]
    fn scaled(value: float32x4_t, scale: Option<float32x4_t>, offset: float32x4_t) -> float32x4_t {
        match scale {
            Some(scale) => vsubq_f32(vmulq_f32(value, scale), offset),
            None => vsubq_f32(value, offset),
        }
    }

    #[target_feature(enable = "neon")]
    fn write_bf16(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, reciprocal: f32) {
        let (chunks, rest) = src.as_chunks::<16>();
        let (out, out_rest) = dst.as_chunks_mut::<32>();
        let scale = (scale != 1.0).then(|| vdupq_n_f32(scale));
        let offset = vdupq_n_f32(offset);
        let reciprocal = vdupq_n_f32(reciprocal);
        let halve = |value: float32x4_t| -> uint32x4_t {
            vreinterpretq_u32_f32(vmulq_f32(scaled(value, scale, offset), reciprocal))
        };
        for (chunk, out) in chunks.iter().zip(out.iter_mut()) {
            // SAFETY: `chunk` is exactly 16 bytes.
            let [a, b, c, d] = widen(unsafe { vld1q_u8(chunk.as_ptr()) });
            let first = vrshrn_high_n_u32::<16>(vrshrn_n_u32::<16>(halve(a)), halve(b));
            let second = vrshrn_high_n_u32::<16>(vrshrn_n_u32::<16>(halve(c)), halve(d));
            // SAFETY: `out` is exactly 32 bytes: two stores of eight `u16`s.
            unsafe {
                vst1q_u16(out.as_mut_ptr().cast::<u16>(), first);
                vst1q_u16(out.as_mut_ptr().add(16).cast::<u16>(), second);
            }
        }
        // The tail: one vector pass per byte, so a tail byte takes the value
        // a chunked byte takes.
        for (out, &value) in out_rest.as_chunks_mut::<2>().0.iter_mut().zip(rest) {
            let mut chunk = [0u8; 16];
            chunk[0] = value;
            let mut word = [0u8; 32];
            write_bf16_one(&mut word, &chunk, scale, offset, reciprocal);
            *out = [word[0], word[1]];
        }
    }

    /// One vector pass over a 16-byte input, for the scalar tail: the same
    /// instructions as the main loop, so a tail byte takes the same value a
    /// chunked byte takes.
    #[inline]
    #[target_feature(enable = "neon")]
    fn write_bf16_one(
        out: &mut [u8; 32],
        chunk: &[u8; 16],
        scale: Option<float32x4_t>,
        offset: float32x4_t,
        reciprocal: float32x4_t,
    ) {
        // SAFETY: `chunk` is exactly 16 bytes.
        let [a, b, _, _] = widen(unsafe { vld1q_u8(chunk.as_ptr()) });
        let halve = |value: float32x4_t| -> uint32x4_t {
            vreinterpretq_u32_f32(vmulq_f32(scaled(value, scale, offset), reciprocal))
        };
        let first = vrshrn_high_n_u32::<16>(vrshrn_n_u32::<16>(halve(a)), halve(b));
        // SAFETY: `out` is 32 bytes; one store of eight `u16`s.
        unsafe { vst1q_u16(out.as_mut_ptr().cast::<u16>(), first) };
    }

    #[target_feature(enable = "neon")]
    fn write_f32(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, divisor: f32) {
        let (chunks, rest) = src.as_chunks::<16>();
        let (out, out_rest) = dst.as_chunks_mut::<64>();
        let scale = (scale != 1.0).then(|| vdupq_n_f32(scale));
        let offset = vdupq_n_f32(offset);
        let divisor = vdupq_n_f32(divisor);
        for (chunk, out) in chunks.iter().zip(out.iter_mut()) {
            // SAFETY: `chunk` is exactly 16 bytes.
            let values = widen(unsafe { vld1q_u8(chunk.as_ptr()) });
            for (quad, value) in values.into_iter().enumerate() {
                let quotient = vdivq_f32(scaled(value, scale, offset), divisor);
                // SAFETY: `out` is exactly 64 bytes: four stores of four `f32`s.
                unsafe { vst1q_f32(out.as_mut_ptr().add(quad * 16).cast::<f32>(), quotient) };
            }
        }
        for (out, &value) in out_rest.as_chunks_mut::<4>().0.iter_mut().zip(rest) {
            let single = [value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            // SAFETY: `single` is exactly 16 bytes.
            let [first, _, _, _] = widen(unsafe { vld1q_u8(single.as_ptr()) });
            let quotient = vdivq_f32(scaled(first, scale, offset), divisor);
            let mut word = [0f32; 4];
            // SAFETY: `word` holds four `f32`s.
            unsafe { vst1q_f32(word.as_mut_ptr(), quotient) };
            *out = word[0].to_le_bytes();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every byte of every channel through tables with 256 distinct entries,
    /// with blocks shorter and longer than a vector and an odd number of
    /// blocks: the written words are the tables' entries in the per-element
    /// order `tables[(index / block) % 3][byte]`.
    #[test]
    fn every_byte_of_every_channel_maps_through_its_table() {
        let tables: [[u16; 256]; 3] = std::array::from_fn(|channel| {
            std::array::from_fn(|value| {
                (value as u16).wrapping_mul(0x9e37) ^ ((channel as u16) << 13)
            })
        });
        let wide: [[u32; 256]; 3] = std::array::from_fn(|channel| {
            std::array::from_fn(|value| {
                (value as u32).wrapping_mul(0x9e37_79b9) ^ ((channel as u32) << 29)
            })
        });
        for block in [1usize, 7, 16, 17, 512] {
            let src: Vec<u8> = (0..3 * block * 3 + 5)
                .map(|i| (i * 131 % 256) as u8)
                .collect();
            let mut dst = vec![0u8; src.len() * 2];
            write_u16(&mut dst, &src, block, &tables);
            let expected: Vec<u8> = src
                .iter()
                .enumerate()
                .flat_map(|(i, &v)| tables[(i / block) % 3][v as usize].to_le_bytes())
                .collect();
            assert_eq!(dst, expected, "u16, block {block}");
            let mut dst = vec![0u8; src.len() * 4];
            write_u32(&mut dst, &src, block, &wide);
            let expected: Vec<u8> = src
                .iter()
                .enumerate()
                .flat_map(|(i, &v)| wide[(i / block) % 3][v as usize].to_le_bytes())
                .collect();
            assert_eq!(dst, expected, "u32, block {block}");
        }
    }

    /// Tables no formula reproduces (arbitrary words) still come out as the
    /// tables: the formula paths decline them.
    #[test]
    fn a_table_the_formula_cannot_reproduce_falls_back_to_the_lookup() {
        let norm = PixelNorm {
            scale: 1.0,
            offset: [127.5; 3],
            divisor: [127.5; 3],
            channel_block: 16,
        };
        let tables: [[u16; 256]; 3] = std::array::from_fn(|channel| {
            std::array::from_fn(|value| (value * 3 + channel) as u16)
        });
        let wide: [[u32; 256]; 3] = std::array::from_fn(|channel| {
            std::array::from_fn(|value| (value * 5 + channel) as u32)
        });
        let src: Vec<u8> = (0..3 * 16 * 2).map(|i| (i * 37 % 256) as u8).collect();
        let mut dst = vec![0u8; src.len() * 2];
        write_bf16(&mut dst, &src, 16, &norm, &tables);
        let expected: Vec<u8> = src
            .iter()
            .enumerate()
            .flat_map(|(i, &v)| tables[(i / 16) % 3][v as usize].to_le_bytes())
            .collect();
        assert_eq!(dst, expected);
        let mut dst = vec![0u8; src.len() * 4];
        write_f32(&mut dst, &src, 16, &norm, &wide);
        let expected: Vec<u8> = src
            .iter()
            .enumerate()
            .flat_map(|(i, &v)| wide[(i / 16) % 3][v as usize].to_le_bytes())
            .collect();
        assert_eq!(dst, expected);
    }

    /// On the target that has the vector path, the normalizations the
    /// pipelines run (mean and std 0.5 with rescale; the CLIP constants with
    /// rescale, both in byte units as the processors build them) take it: a
    /// rounding regression in the formula would not change the bytes (the
    /// table lookup is the fallback) but would lose the speed without a
    /// failing test, so this one fails instead.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_vector_formula_reproduces_the_tables_of_the_pipelines_normalizations() {
        use crate::encoder_inputs::f32_to_bf16_bits;

        let into_bytes = (1.0_f64 / (1.0 / 255.0)) as f32;
        let norms = [
            PixelNorm {
                scale: 1.0,
                offset: [0.5_f64 as f32 * into_bytes; 3],
                divisor: [0.5_f64 as f32 * into_bytes; 3],
                channel_block: 512,
            },
            PixelNorm {
                scale: 1.0,
                offset: [
                    0.481_454_66_f64 as f32 * into_bytes,
                    0.457_827_5_f64 as f32 * into_bytes,
                    0.408_210_73_f64 as f32 * into_bytes,
                ],
                divisor: [
                    0.268_629_54_f64 as f32 * into_bytes,
                    0.261_302_58_f64 as f32 * into_bytes,
                    0.275_777_11_f64 as f32 * into_bytes,
                ],
                channel_block: 392,
            },
        ];
        for norm in norms {
            let bf16 = norm
                .table_f32()
                .map(|channel| channel.map(f32_to_bf16_bits));
            assert!(
                neon::Bf16Formula::reproducing(&norm, &bf16).is_some(),
                "bf16 formula declined {norm:?}"
            );
            let f32s = norm.table_f32().map(|channel| channel.map(f32::to_bits));
            assert!(
                neon::F32Formula::reproducing(&norm, &f32s).is_some(),
                "f32 formula declined {norm:?}"
            );
        }
    }
}
