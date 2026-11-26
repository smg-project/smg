//! Pixel bytes to wire words: every byte of a channel maps to the bits of its
//! wire value, `PixelNorm::table_f32` (the reference processors' per-element
//! arithmetic) rounded to the wire's float width. The tables are the single
//! source of the values. This module moves them: a scalar table lookup per
//! channel block on every target, and on AArch64 and x86-64 a vector formula
//! that is used for a write only after it has reproduced that write's tables
//! for all 256 bytes of all three channels, so the bytes written are the
//! tables' whichever path runs.
//!
//! A reference processor that normalizes in `bfloat16`, rounding after every
//! step, has values no `PixelNorm` gives: [`Bf16Steps`] is that arithmetic
//! and [`StepsWriter`] moves its table the same way, with a vector formula
//! of its own on x86-64 and the lookup on every other target.
//!
//! Why not a vector table lookup: a 256-entry table is 512 bytes per output
//! byte plane, and the vector unit's byte permutes index 64 bytes at a time,
//! which leaves such a lookup slower than the scalar loop; the arithmetic is
//! a handful of lane-wise operations per 16 pixels.
//!
//! The AArch64 path is the crate's only use of `core::arch` vector loads and
//! stores, hence the local override of the workspace-wide
//! `unsafe_code = "deny"`; every pointer it forms comes from an exact-size
//! chunk of a slice. The x86-64 path carries no `unsafe`: `multiversion`
//! selects its AVX2 clone at the first call, and every vector word enters and
//! leaves it through a `bytemuck` cast of an exact-size byte array.
#![cfg_attr(target_arch = "aarch64", allow(unsafe_code))]

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use formula::{Bf16Formula, F32Formula};

#[cfg(target_arch = "x86_64")]
use self::avx2 as vector;
#[cfg(target_arch = "aarch64")]
use self::neon as vector;
use crate::encoder_inputs::{f32_to_bf16_bits, f32_to_f16_bits, EncoderDtype, PixelNorm};

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
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if let Some(formula) = Bf16Formula::reproducing(norm, tables) {
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
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if let Some(formula) = F32Formula::reproducing(norm, tables) {
        debug_assert_eq!(dst.len(), src.len() * 4);
        for ((dst, src), channel) in channel_blocks(dst, src, block, 4).zip((0..3).cycle()) {
            formula.write(dst, src, channel);
        }
        return;
    }
    let _ = norm;
    write_u32(dst, src, block, tables);
}

/// The converter of one normalization to one wire dtype, decided once per
/// image for the one-pass writers (the patch layout and the normalization
/// together): the vector formula where it reproduces the tables, the tables
/// otherwise. `None` for the `uint8` wire, which has nothing to convert.
pub(crate) enum WireWriter {
    Half {
        tables: Box<[[u16; 256]; 3]>,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        formula: Option<Bf16Formula>,
    },
    Single {
        tables: Box<[[u32; 256]; 3]>,
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        formula: Option<F32Formula>,
    },
}

impl WireWriter {
    pub(crate) fn new(norm: &PixelNorm, dtype: EncoderDtype) -> Option<Self> {
        match dtype {
            EncoderDtype::Uint8 => None,
            EncoderDtype::Float32 => {
                let tables = Box::new(norm.table_f32().map(|channel| channel.map(f32::to_bits)));
                Some(Self::Single {
                    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                    formula: F32Formula::reproducing(norm, &tables),
                    tables,
                })
            }
            EncoderDtype::BFloat16 => {
                let tables = Box::new(
                    norm.table_f32()
                        .map(|channel| channel.map(f32_to_bf16_bits)),
                );
                Some(Self::Half {
                    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                    formula: Bf16Formula::reproducing(norm, &tables),
                    tables,
                })
            }
            EncoderDtype::Float16 => Some(Self::Half {
                tables: Box::new(norm.table_f32().map(|channel| channel.map(f32_to_f16_bits))),
                // The half-up rounding of the vector formula is the
                // bfloat16's; float16 takes the tables.
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                formula: None,
            }),
        }
    }

    /// The writer of `table` narrowed to `dtype`, through the tables alone:
    /// for values no [`PixelNorm`] describes.
    fn of_table(table: &[[f32; 256]; 3], dtype: EncoderDtype) -> Option<Self> {
        let half = |convert: fn(f32) -> u16| Self::Half {
            tables: Box::new(table.map(|channel| channel.map(convert))),
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            formula: None,
        };
        match dtype {
            EncoderDtype::Uint8 => None,
            EncoderDtype::Float32 => Some(Self::Single {
                tables: Box::new(table.map(|channel| channel.map(f32::to_bits))),
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                formula: None,
            }),
            EncoderDtype::BFloat16 => Some(half(f32_to_bf16_bits)),
            EncoderDtype::Float16 => Some(half(f32_to_f16_bits)),
        }
    }

    /// Bytes per written element.
    pub(crate) fn element_size(&self) -> usize {
        match self {
            Self::Half { .. } => 2,
            Self::Single { .. } => 4,
        }
    }

    /// Writes one channel's run: `src.len()` pixel bytes of `channel` as
    /// `dst` (`element_size()` bytes each).
    pub(crate) fn write_run(&self, dst: &mut [u8], src: &[u8], channel: usize) {
        debug_assert_eq!(dst.len(), src.len() * self.element_size());
        match self {
            Self::Half { tables, .. } => {
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                if let Self::Half {
                    formula: Some(formula),
                    ..
                } = self
                {
                    formula.write(dst, src, channel);
                    return;
                }
                let table = &tables[channel];
                for (d, &v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                    *d = table[v as usize].to_le_bytes();
                }
            }
            Self::Single { tables, .. } => {
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                if let Self::Single {
                    formula: Some(formula),
                    ..
                } = self
                {
                    formula.write(dst, src, channel);
                    return;
                }
                let table = &tables[channel];
                for (d, &v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                    *d = table[v as usize].to_le_bytes();
                }
            }
        }
    }

    /// [`Self::write_run`] for one channel of interleaved RGB pixels: the
    /// `channel` byte of every pixel as `dst`, through the tables (a strided
    /// source has no vector formula).
    pub(crate) fn write_channel(&self, dst: &mut [u8], pixels: &[[u8; 3]], channel: usize) {
        debug_assert_eq!(dst.len(), pixels.len() * self.element_size());
        match self {
            Self::Half { tables, .. } => {
                let table = &tables[channel];
                for (d, pixel) in dst.as_chunks_mut::<2>().0.iter_mut().zip(pixels) {
                    *d = table[usize::from(pixel[channel])].to_le_bytes();
                }
            }
            Self::Single { tables, .. } => {
                let table = &tables[channel];
                for (d, pixel) in dst.as_chunks_mut::<4>().0.iter_mut().zip(pixels) {
                    *d = table[usize::from(pixel[channel])].to_le_bytes();
                }
            }
        }
    }
}

/// The arithmetic of a reference processor that rescales and normalizes in
/// `bfloat16`, rounding after every step: `byte * factor`, `- mean[c]`,
/// `/ std[c]`. [`PixelNorm`] fuses the three and rounds once, in `f32`, which
/// is a neighbouring `bfloat16` for many bytes (half of them at a mean and
/// std of 0.5), so such a processor's floats of every dtype are cut from
/// [`Self::table_f32`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bf16Steps {
    /// Multiplies the byte, as an `f32` scalar.
    pub(crate) factor: f32,
    /// Subtracted next, per channel, as its nearest `bfloat16`.
    pub(crate) mean: [f32; 3],
    /// Divides last, per channel, as its nearest `bfloat16`.
    pub(crate) std: [f32; 3],
}

impl Bf16Steps {
    /// The value each byte maps to, per channel: a `bfloat16` held in an
    /// `f32`.
    pub(crate) fn table_f32(&self) -> [[f32; 256]; 3] {
        let channel = |mean: f32, std: f32| -> [f32; 256] {
            std::array::from_fn(|value| {
                to_bf16(to_bf16(to_bf16(value as f32 * self.factor) - mean) / std)
            })
        };
        let mean = self.mean.map(to_bf16);
        let std = self.std.map(to_bf16);
        let mut table = [channel(mean[0], std[0]); 3];
        for later in 1..3 {
            // A channel of the first one's mean and std has its values.
            if (mean[later].to_bits(), std[later].to_bits())
                != (mean[0].to_bits(), std[0].to_bits())
            {
                table[later] = channel(mean[later], std[later]);
            }
        }
        table
    }
}

/// `value` rounded to the nearest `bfloat16` (ties to even), still an `f32`.
fn to_bf16(value: f32) -> f32 {
    f32::from_bits(u32::from(f32_to_bf16_bits(value)) << 16)
}

/// [`WireWriter`] for [`Bf16Steps`]: on x86-64 the vector formula where it
/// reproduces the steps' values, else (and on every other target) the tables
/// of those values in the wire's dtype.
pub(crate) enum StepsWriter {
    Tables(WireWriter),
    #[cfg(target_arch = "x86_64")]
    Formula(StepsFormula),
}

impl StepsWriter {
    /// `None` for the `uint8` wire, as [`WireWriter::new`].
    pub(crate) fn new(steps: &Bf16Steps, dtype: EncoderDtype) -> Option<Self> {
        let table = steps.table_f32();
        #[cfg(target_arch = "x86_64")]
        if let Some(formula) = StepsFormula::reproducing(steps, &table, dtype) {
            return Some(Self::Formula(formula));
        }
        WireWriter::of_table(&table, dtype).map(Self::Tables)
    }

    /// Bytes per written element.
    pub(crate) fn element_size(&self) -> usize {
        match self {
            Self::Tables(tables) => tables.element_size(),
            #[cfg(target_arch = "x86_64")]
            Self::Formula(formula) => formula.element_size(),
        }
    }

    /// [`WireWriter::write_run`] with the steps' values.
    pub(crate) fn write_run(&self, dst: &mut [u8], src: &[u8], channel: usize) {
        debug_assert_eq!(dst.len(), src.len() * self.element_size());
        match self {
            Self::Tables(tables) => tables.write_run(dst, src, channel),
            #[cfg(target_arch = "x86_64")]
            Self::Formula(formula) => formula.write(dst, src, channel),
        }
    }
}

/// The vector formula of [`Bf16Steps`]: `r(r(byte * (factor / std)) - mean /
/// std)`, `r` rounding to `bfloat16` half-up on the bit pattern. Where the
/// std is a power of two the division commutes with every rounding and the
/// three steps collapse to these two. Neither that nor the absence of a tie
/// is assumed: the formula is used only after it has written the table's 768
/// values, checked with the code that writes.
#[cfg(target_arch = "x86_64")]
pub(crate) struct StepsFormula {
    factor: [f32; 3],
    mean: [f32; 3],
    /// `f32` words (each `bfloat16` widened) rather than `bfloat16` words.
    single: bool,
}

#[cfg(target_arch = "x86_64")]
impl StepsFormula {
    /// The formula for `steps` in `dtype`, if it reproduces `table` for every
    /// byte of every channel.
    fn reproducing(
        steps: &Bf16Steps,
        table: &[[f32; 256]; 3],
        dtype: EncoderDtype,
    ) -> Option<Self> {
        let single = match dtype {
            EncoderDtype::BFloat16 => false,
            EncoderDtype::Float32 => true,
            // A `float16` word is not the `bfloat16`'s bits: the tables
            // write it.
            EncoderDtype::Float16 | EncoderDtype::Uint8 => return None,
        };
        if !avx2::detected() {
            return None;
        }
        let std = steps.std.map(to_bf16);
        let formula = Self {
            factor: std.map(|std| steps.factor / std),
            mean: std::array::from_fn(|channel| to_bf16(steps.mean[channel]) / std[channel]),
            single,
        };
        let input: [u8; 256] = std::array::from_fn(|value| value as u8);
        let mut output = [0u8; 1024];
        let output = &mut output[..input.len() * formula.element_size()];
        (0..3)
            .all(|channel| {
                formula.write(output, &input, channel);
                let values = table[channel].iter().map(|value| value.to_bits());
                // Words of a width the compiler knows: a slice of a width
                // chosen here compares through a library call per value.
                if single {
                    let (written, _) = output.as_chunks::<4>();
                    written
                        .iter()
                        .zip(values)
                        .all(|(written, value)| *written == value.to_le_bytes())
                } else {
                    // A `bfloat16` word is the high half of its `f32`.
                    let (written, _) = output.as_chunks::<2>();
                    written
                        .iter()
                        .zip(values)
                        .all(|(written, value)| *written == ((value >> 16) as u16).to_le_bytes())
                }
            })
            .then_some(formula)
    }

    /// Bytes per written element.
    fn element_size(&self) -> usize {
        if self.single {
            4
        } else {
            2
        }
    }

    /// Writes `channel`'s words for `src` into `dst`.
    fn write(&self, dst: &mut [u8], src: &[u8], channel: usize) {
        let (factor, mean) = (self.factor[channel], self.mean[channel]);
        if self.single {
            avx2::write_f32_steps(dst, src, factor, mean);
        } else {
            avx2::write_bf16_steps(dst, src, factor, mean);
        }
    }
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

/// The vector formula of a normalization: the constants per channel and the
/// check against the tables, shared by the two architectures' paths; the
/// `vector` module runs the arithmetic.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod formula {
    use super::{vector, PixelNorm};

    /// The bytes 0..=255 in order: the input that produces a whole table.
    fn every_byte() -> [u8; 256] {
        std::array::from_fn(|value| value as u8)
    }

    /// `(byte * scale - offset) * (1 / divisor)` in `f32`, rounded to
    /// `bfloat16` half-up on the bit pattern: cheaper than a division and
    /// than round-to-nearest-even, and exactly the table's value whenever
    /// [`Self::reproducing`] says so, which is the only time it runs.
    pub(crate) struct Bf16Formula {
        scale: f32,
        offset: [f32; 3],
        reciprocal: [f32; 3],
    }

    impl Bf16Formula {
        /// The formula for `norm`, if it reproduces `tables` for every byte of
        /// every channel (768 values, checked with the same code that writes).
        pub(crate) fn reproducing(norm: &PixelNorm, tables: &[[u16; 256]; 3]) -> Option<Self> {
            if !vector::detected() {
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
        pub(crate) fn write(&self, dst: &mut [u8], src: &[u8], channel: usize) {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON was detected in `reproducing`, the only constructor.
            unsafe {
                vector::write_bf16(
                    dst,
                    src,
                    self.scale,
                    self.offset[channel],
                    self.reciprocal[channel],
                );
            }
            #[cfg(target_arch = "x86_64")]
            vector::write_bf16(
                dst,
                src,
                self.scale,
                self.offset[channel],
                self.reciprocal[channel],
            );
        }
    }

    /// `(byte * scale - offset) / divisor` in `f32` on the vector unit.
    pub(crate) struct F32Formula {
        scale: f32,
        offset: [f32; 3],
        divisor: [f32; 3],
    }

    impl F32Formula {
        /// The formula for `norm`, if it reproduces `tables` for every byte of
        /// every channel.
        pub(crate) fn reproducing(norm: &PixelNorm, tables: &[[u32; 256]; 3]) -> Option<Self> {
            if !vector::detected() {
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
        pub(crate) fn write(&self, dst: &mut [u8], src: &[u8], channel: usize) {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON was detected in `reproducing`, the only constructor.
            unsafe {
                vector::write_f32(
                    dst,
                    src,
                    self.scale,
                    self.offset[channel],
                    self.divisor[channel],
                );
            }
            #[cfg(target_arch = "x86_64")]
            vector::write_f32(
                dst,
                src,
                self.scale,
                self.offset[channel],
                self.divisor[channel],
            );
        }
    }
}

/// The AArch64 path: the formula on four `f32` lanes with NEON.
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        float32x4_t, uint32x4_t, uint8x16_t, vcvtq_f32_u32, vdivq_f32, vdupq_n_f32, vget_low_u16,
        vget_low_u8, vld1q_u8, vmovl_high_u16, vmovl_high_u8, vmovl_u16, vmovl_u8, vmulq_f32,
        vreinterpretq_u32_f32, vrshrn_high_n_u32, vrshrn_n_u32, vst1q_f32, vst1q_u16, vsubq_f32,
    };

    /// Whether this core has the vector unit the formula runs on.
    pub(super) fn detected() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
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

    /// `channel`'s `bfloat16` words (two bytes each) for the pixel bytes of
    /// `src` into `dst`.
    #[target_feature(enable = "neon")]
    pub(super) fn write_bf16(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, reciprocal: f32) {
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

    /// `channel`'s `f32` words (four bytes each) for the pixel bytes of `src`
    /// into `dst`.
    #[target_feature(enable = "neon")]
    pub(super) fn write_f32(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, divisor: f32) {
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

/// The x86-64 path: the formula on eight `f32` lanes with AVX2. `multiversion`
/// compiles each entry twice, with AVX2 and without, and selects the clone at
/// the first call; the clone without AVX2 is the same arithmetic one value at
/// a time, reached only through a formula built without the detection, which
/// none is. No `unsafe`: the words move through `bytemuck` casts of
/// exact-size arrays.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::{
        __m128i, __m256, __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256,
        _mm256_blend_epi16, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cvtepi32_ps,
        _mm256_cvtepu8_epi16, _mm256_cvtepu8_epi32, _mm256_div_ps, _mm256_mul_ps,
        _mm256_packus_epi32, _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_set1_ps,
        _mm256_srli_epi32, _mm256_sub_ps, _mm_srli_si128,
    };

    use multiversion::{multiversion, target::match_target};

    /// Whether this core has the vector unit the formula runs on.
    pub(super) fn detected() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
    }

    /// `channel`'s `bfloat16` words (two bytes each) for the pixel bytes of
    /// `src` into `dst`.
    #[multiversion(targets("x86_64+avx2"))]
    pub(super) fn write_bf16(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, reciprocal: f32) {
        match_target! {
            "x86_64+avx2" => write_bf16_avx2(dst, src, scale, offset, reciprocal),
            _ => write_bf16_scalar(dst, src, scale, offset, reciprocal),
        }
    }

    /// `channel`'s `f32` words (four bytes each) for the pixel bytes of `src`
    /// into `dst`.
    #[multiversion(targets("x86_64+avx2"))]
    pub(super) fn write_f32(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, divisor: f32) {
        match_target! {
            "x86_64+avx2" => write_f32_avx2(dst, src, scale, offset, divisor),
            _ => write_f32_scalar(dst, src, scale, offset, divisor),
        }
    }

    /// 16 bytes as two vectors of eight `f32`s.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn widen(bytes: [u8; 16]) -> [__m256; 2] {
        let bytes: __m128i = bytemuck::must_cast(bytes);
        [
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)),
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(bytes))),
        ]
    }

    /// `value * scale - offset`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn scaled(value: __m256, scale: __m256, offset: __m256) -> __m256 {
        _mm256_sub_ps(_mm256_mul_ps(value, scale), offset)
    }

    /// Eight `f32`s to their `bfloat16` bits, half-up on the bit pattern: the
    /// bias of `0x8000` wraps the way a rounding narrowing shift's wider sum
    /// does, since only the low sixteen bits of the shifted sum are kept.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn halve(value: __m256, bias: __m256i) -> __m256i {
        _mm256_srli_epi32::<16>(_mm256_add_epi32(_mm256_castps_si256(value), bias))
    }

    /// The thirty-two bytes of sixteen `bfloat16` words for sixteen pixel
    /// bytes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn bf16_words(
        chunk: [u8; 16],
        scale: __m256,
        offset: __m256,
        reciprocal: __m256,
        bias: __m256i,
    ) -> [u8; 32] {
        let [low, high] = widen(chunk);
        let low = halve(_mm256_mul_ps(scaled(low, scale, offset), reciprocal), bias);
        let high = halve(_mm256_mul_ps(scaled(high, scale, offset), reciprocal), bias);
        // The pack interleaves the two 128-bit halves (words 0-3, 8-11, 4-7,
        // 12-15); the permute puts them in order.
        bytemuck::must_cast(_mm256_permute4x64_epi64::<0b1101_1000>(
            _mm256_packus_epi32(low, high),
        ))
    }

    /// The sixty-four bytes of sixteen `f32` words for sixteen pixel bytes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn f32_words(chunk: [u8; 16], scale: __m256, offset: __m256, divisor: __m256) -> [u8; 64] {
        let [low, high] = widen(chunk);
        bytemuck::must_cast([
            _mm256_div_ps(scaled(low, scale, offset), divisor),
            _mm256_div_ps(scaled(high, scale, offset), divisor),
        ])
    }

    #[target_feature(enable = "avx2")]
    fn write_bf16_avx2(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, reciprocal: f32) {
        let (chunks, rest) = src.as_chunks::<16>();
        let (out, out_rest) = dst.as_chunks_mut::<32>();
        let scale = _mm256_set1_ps(scale);
        let offset = _mm256_set1_ps(offset);
        let reciprocal = _mm256_set1_ps(reciprocal);
        let bias = _mm256_set1_epi32(0x8000);
        for (chunk, out) in chunks.iter().zip(out.iter_mut()) {
            *out = bf16_words(*chunk, scale, offset, reciprocal, bias);
        }
        // The tail: the same vector pass over a zero-padded chunk, so a tail
        // byte takes the value a chunked byte takes.
        if !rest.is_empty() {
            let mut chunk = [0u8; 16];
            chunk[..rest.len()].copy_from_slice(rest);
            let words = bf16_words(chunk, scale, offset, reciprocal, bias);
            for (out, word) in out_rest
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(words.as_chunks::<2>().0)
            {
                *out = *word;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    fn write_f32_avx2(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, divisor: f32) {
        let (chunks, rest) = src.as_chunks::<16>();
        let (out, out_rest) = dst.as_chunks_mut::<64>();
        let scale = _mm256_set1_ps(scale);
        let offset = _mm256_set1_ps(offset);
        let divisor = _mm256_set1_ps(divisor);
        for (chunk, out) in chunks.iter().zip(out.iter_mut()) {
            *out = f32_words(*chunk, scale, offset, divisor);
        }
        if !rest.is_empty() {
            let mut chunk = [0u8; 16];
            chunk[..rest.len()].copy_from_slice(rest);
            let words = f32_words(chunk, scale, offset, divisor);
            for (out, word) in out_rest
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(words.as_chunks::<4>().0)
            {
                *out = *word;
            }
        }
    }

    /// The formula one value at a time, in the same `f32` operations and the
    /// same rounding: the clone for a core without AVX2.
    fn write_bf16_scalar(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, reciprocal: f32) {
        for (out, &value) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
            let bits = ((f32::from(value) * scale - offset) * reciprocal).to_bits();
            *out = ((bits.wrapping_add(0x8000) >> 16) as u16).to_le_bytes();
        }
    }

    fn write_f32_scalar(dst: &mut [u8], src: &[u8], scale: f32, offset: f32, divisor: f32) {
        for (out, &value) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
            *out = ((f32::from(value) * scale - offset) / divisor).to_le_bytes();
        }
    }

    /// [`write_bf16`] for the stepped arithmetic: the `bfloat16` words of
    /// `byte * factor - mean`, the product rounded to `bfloat16` before the
    /// mean comes off.
    #[multiversion(targets("x86_64+avx2"))]
    pub(super) fn write_bf16_steps(dst: &mut [u8], src: &[u8], factor: f32, mean: f32) {
        match_target! {
            "x86_64+avx2" => write_bf16_steps_avx2(dst, src, factor, mean),
            _ => write_bf16_steps_scalar(dst, src, factor, mean),
        }
    }

    /// [`write_bf16_steps`] with each word widened to its `f32` (four bytes
    /// each).
    #[multiversion(targets("x86_64+avx2"))]
    pub(super) fn write_f32_steps(dst: &mut [u8], src: &[u8], factor: f32, mean: f32) {
        match_target! {
            "x86_64+avx2" => write_f32_steps_avx2(dst, src, factor, mean),
            _ => write_f32_steps_scalar(dst, src, factor, mean),
        }
    }

    /// The bits of `value * factor - mean` for eight `f32`s with the
    /// `bfloat16` bias added, the product rounded to `bfloat16` on the way:
    /// the same bias rounds it half-up and `word` clears the half a
    /// `bfloat16` drops. The high half of each lane is the result's word.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn steps_bits(
        value: __m256,
        factor: __m256,
        mean: __m256,
        bias: __m256i,
        word: __m256i,
    ) -> __m256i {
        let product = _mm256_add_epi32(_mm256_castps_si256(_mm256_mul_ps(value, factor)), bias);
        let product = _mm256_castsi256_ps(_mm256_and_si256(product, word));
        _mm256_add_epi32(_mm256_castps_si256(_mm256_sub_ps(product, mean)), bias)
    }

    #[target_feature(enable = "avx2")]
    fn write_bf16_steps_avx2(dst: &mut [u8], src: &[u8], factor: f32, mean: f32) {
        let (chunks, rest) = src.as_chunks::<16>();
        let (out, out_rest) = dst.as_chunks_mut::<32>();
        let factor = _mm256_set1_ps(factor);
        let mean = _mm256_set1_ps(mean);
        let bias = _mm256_set1_epi32(0x8000);
        let word = _mm256_set1_epi32(-0x1_0000);
        let words = |chunk: [u8; 16]| -> [u8; 32] {
            // A 32-bit lane takes two neighbouring bytes, the even one in its
            // low half. Each is converted in its own vector; the even one's
            // word then moves down and the odd one's stands where it goes, so
            // a blend puts the sixteen in order with no pack and no permute.
            let pairs = _mm256_cvtepu8_epi16(bytemuck::must_cast(chunk));
            let even = _mm256_cvtepi32_ps(_mm256_andnot_si256(word, pairs));
            let odd = _mm256_cvtepi32_ps(_mm256_srli_epi32::<16>(pairs));
            let even = _mm256_srli_epi32::<16>(steps_bits(even, factor, mean, bias, word));
            let odd = steps_bits(odd, factor, mean, bias, word);
            bytemuck::must_cast(_mm256_blend_epi16::<0b1010_1010>(even, odd))
        };
        for (chunk, out) in chunks.iter().zip(out.iter_mut()) {
            *out = words(*chunk);
        }
        if !rest.is_empty() {
            let mut chunk = [0u8; 16];
            chunk[..rest.len()].copy_from_slice(rest);
            let tail = words(chunk);
            for (out, word) in out_rest
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(tail.as_chunks::<2>().0)
            {
                *out = *word;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    fn write_f32_steps_avx2(dst: &mut [u8], src: &[u8], factor: f32, mean: f32) {
        let (chunks, rest) = src.as_chunks::<16>();
        let (out, out_rest) = dst.as_chunks_mut::<64>();
        let factor = _mm256_set1_ps(factor);
        let mean = _mm256_set1_ps(mean);
        let bias = _mm256_set1_epi32(0x8000);
        let word = _mm256_set1_epi32(-0x1_0000);
        let words = |chunk: [u8; 16]| -> [u8; 64] {
            let [low, high] = widen(chunk);
            bytemuck::must_cast([
                _mm256_and_si256(steps_bits(low, factor, mean, bias, word), word),
                _mm256_and_si256(steps_bits(high, factor, mean, bias, word), word),
            ])
        };
        for (chunk, out) in chunks.iter().zip(out.iter_mut()) {
            *out = words(*chunk);
        }
        if !rest.is_empty() {
            let mut chunk = [0u8; 16];
            chunk[..rest.len()].copy_from_slice(rest);
            let tail = words(chunk);
            for (out, word) in out_rest
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(tail.as_chunks::<4>().0)
            {
                *out = *word;
            }
        }
    }

    /// [`steps_bits`] one value at a time, in the same `f32` operations and
    /// the same rounding: the clone for a core without AVX2.
    fn steps_bits_scalar(value: u8, factor: f32, mean: f32) -> u32 {
        let product = (f32::from(value) * factor).to_bits().wrapping_add(0x8000) & 0xffff_0000;
        (f32::from_bits(product) - mean)
            .to_bits()
            .wrapping_add(0x8000)
    }

    fn write_bf16_steps_scalar(dst: &mut [u8], src: &[u8], factor: f32, mean: f32) {
        for (out, &value) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
            *out = ((steps_bits_scalar(value, factor, mean) >> 16) as u16).to_le_bytes();
        }
    }

    fn write_f32_steps_scalar(dst: &mut [u8], src: &[u8], factor: f32, mean: f32) {
        for (out, &value) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
            *out = (steps_bits_scalar(value, factor, mean) & 0xffff_0000).to_le_bytes();
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

    /// One channel of interleaved pixels takes the values its bytes take as
    /// a run: every byte of every channel, in every float dtype, into a
    /// buffer that held other bytes.
    #[test]
    fn a_channel_of_interleaved_pixels_is_written_as_its_run() {
        let norm = PixelNorm {
            scale: 1.0,
            offset: [123.675, 116.28, 103.53],
            divisor: [58.395, 57.12, 57.375],
            channel_block: 16,
        };
        let pixels: Vec<[u8; 3]> = (0..=255u8)
            .map(|value| [value, value.wrapping_mul(7), 255 - value])
            .collect();
        for dtype in [
            EncoderDtype::BFloat16,
            EncoderDtype::Float16,
            EncoderDtype::Float32,
        ] {
            let writer = WireWriter::new(&norm, dtype).unwrap();
            for channel in 0..3 {
                let run: Vec<u8> = pixels.iter().map(|pixel| pixel[channel]).collect();
                let mut expected = vec![0u8; run.len() * writer.element_size()];
                writer.write_run(&mut expected, &run, channel);
                let mut written = vec![0xAAu8; expected.len()];
                writer.write_channel(&mut written, &pixels, channel);
                assert_eq!(written, expected, "{dtype}, channel {channel}");
            }
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
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn the_vector_formula_reproduces_the_tables_of_the_pipelines_normalizations() {
        if !vector::detected() {
            return;
        }

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
                Bf16Formula::reproducing(&norm, &bf16).is_some(),
                "bf16 formula declined {norm:?}"
            );
            let f32s = norm.table_f32().map(|channel| channel.map(f32::to_bits));
            assert!(
                F32Formula::reproducing(&norm, &f32s).is_some(),
                "f32 formula declined {norm:?}"
            );
        }
    }

    /// On the target that has the vector path, runs of every length from 0 to
    /// 40 bytes (whole vectors and every tail) through a normalization the
    /// formula reproduces write the tables' words, in both wire widths, with
    /// channel blocks shorter and longer than a vector.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn every_run_length_takes_the_tables_values_on_the_vector_path() {
        if !vector::detected() {
            return;
        }
        let into_bytes = (1.0_f64 / (1.0 / 255.0)) as f32;
        let norm = PixelNorm {
            scale: 1.0,
            offset: [0.5_f64 as f32 * into_bytes; 3],
            divisor: [0.5_f64 as f32 * into_bytes; 3],
            channel_block: 7,
        };
        let bf16 = norm
            .table_f32()
            .map(|channel| channel.map(f32_to_bf16_bits));
        let f32s = norm.table_f32().map(|channel| channel.map(f32::to_bits));
        assert!(Bf16Formula::reproducing(&norm, &bf16).is_some());
        assert!(F32Formula::reproducing(&norm, &f32s).is_some());
        for block in [7usize, 40] {
            for n in 0..=40 {
                let src: Vec<u8> = (0..n).map(|i| (i * 131 % 256) as u8).collect();
                let mut dst = vec![0u8; n * 2];
                write_bf16(&mut dst, &src, block, &norm, &bf16);
                let expected: Vec<u8> = src
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &v)| bf16[(i / block) % 3][v as usize].to_le_bytes())
                    .collect();
                assert_eq!(dst, expected, "bf16, block {block}, {n} bytes");
                let mut dst = vec![0u8; n * 4];
                write_f32(&mut dst, &src, block, &norm, &f32s);
                let expected: Vec<u8> = src
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &v)| f32s[(i / block) % 3][v as usize].to_le_bytes())
                    .collect();
                assert_eq!(dst, expected, "f32, block {block}, {n} bytes");
            }
        }
    }

    /// A rescale by 1/255, then a mean and std of 0.5.
    fn half_steps() -> Bf16Steps {
        Bf16Steps {
            factor: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }

    /// `table`'s words for `src` in `dtype`, one value at a time.
    fn table_words(table: &[f32; 256], src: &[u8], dtype: EncoderDtype) -> Vec<u8> {
        src.iter()
            .flat_map(|&byte| {
                let value = table[usize::from(byte)];
                match dtype {
                    EncoderDtype::BFloat16 => f32_to_bf16_bits(value).to_le_bytes().to_vec(),
                    EncoderDtype::Float16 => f32_to_f16_bits(value).to_le_bytes().to_vec(),
                    _ => value.to_le_bytes().to_vec(),
                }
            })
            .collect()
    }

    /// The steps round after every operation: every value is a `bfloat16`,
    /// byte 128 is one whole `bfloat16` step above zero where the fused
    /// formula gives half a byte step, and half of the 768 words differ from
    /// the fused formula's.
    #[test]
    fn the_steps_table_is_bfloat16_arithmetic_step_by_step() {
        let table = half_steps().table_f32();
        assert_eq!(table[0][0], -1.0);
        assert_eq!(table[0][128], 0.007_812_5);
        assert_eq!(table[0][255], 1.0);
        assert!(table
            .iter()
            .flatten()
            .all(|value| value.to_bits().trailing_zeros() >= 16));

        let fused = PixelNorm {
            scale: 1.0,
            offset: [127.5; 3],
            divisor: [127.5; 3],
            channel_block: 1,
        }
        .table_f32();
        assert_eq!(fused[0][128], 0.5 / 127.5);
        let differing = table
            .iter()
            .flatten()
            .zip(fused.iter().flatten())
            .filter(|(steps, fused)| f32_to_bf16_bits(**steps) != f32_to_bf16_bits(**fused))
            .count();
        assert_eq!(differing, 384);

        // A channel of the first one's mean and std holds the first one's
        // values; a channel of its own mean holds its own.
        let table = Bf16Steps {
            mean: [0.5, 0.25, 0.5],
            ..half_steps()
        }
        .table_f32();
        assert_eq!(table[2], table[0]);
        assert_eq!((table[1][0], table[1][255]), (-0.5, 1.5));
    }

    /// On x86-64 the stepped formula reproduces the table of a std that is a
    /// power of two, for the `bfloat16` and the `f32` wire, and a run of any
    /// length (whole vectors, every tail, all 256 bytes) takes the table's
    /// words through it.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_stepped_formula_writes_the_tables_words_at_every_run_length() {
        if !avx2::detected() {
            return;
        }
        let steps = half_steps();
        let table = steps.table_f32();
        for dtype in [EncoderDtype::BFloat16, EncoderDtype::Float32] {
            let writer = StepsWriter::new(&steps, dtype).unwrap();
            assert!(
                matches!(writer, StepsWriter::Formula(_)),
                "{dtype}: no formula"
            );
            for len in (0..=67).chain([256, 259]) {
                let src: Vec<u8> = (0..len).map(|i: usize| (i * 37 + len) as u8).collect();
                let mut dst = vec![0xa5; len * writer.element_size()];
                writer.write_run(&mut dst, &src, 1);
                assert_eq!(
                    dst,
                    table_words(&table[1], &src, dtype),
                    "{dtype}, {len} bytes"
                );
            }
        }
    }

    /// A block that is not its run's length is refused on either arm: the
    /// vector formula would lay its tail behind the block's last whole
    /// vector, not behind the words it wrote.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "left == right")]
    fn the_steps_writer_refuses_a_block_that_is_not_its_runs_length() {
        let writer = StepsWriter::new(&half_steps(), EncoderDtype::BFloat16).unwrap();
        let src = [0u8; 17];
        let mut dst = vec![0; (src.len() + 16) * writer.element_size()];
        writer.write_run(&mut dst, &src, 1);
    }

    /// Steps the formula does not reproduce (a std that is no power of two)
    /// and the `float16` wire, whose words are not the `bfloat16`'s, are
    /// written through the tables; the `uint8` wire has no writer.
    #[test]
    fn steps_without_a_formula_are_written_through_the_tables() {
        let uneven = Bf16Steps {
            factor: 1.0 / 255.0,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        };
        let src: Vec<u8> = (0..=255).collect();
        for (steps, dtype) in [
            (uneven, EncoderDtype::BFloat16),
            (uneven, EncoderDtype::Float32),
            (uneven, EncoderDtype::Float16),
            (half_steps(), EncoderDtype::Float16),
        ] {
            let writer = StepsWriter::new(&steps, dtype).unwrap();
            // The only form a writer has on another target.
            #[cfg(target_arch = "x86_64")]
            assert!(
                matches!(writer, StepsWriter::Tables(_)),
                "{dtype}: a formula"
            );
            let table = steps.table_f32();
            for (channel, table) in table.iter().enumerate() {
                let mut dst = vec![0xa5; src.len() * writer.element_size()];
                writer.write_run(&mut dst, &src, channel);
                assert_eq!(
                    dst,
                    table_words(table, &src, dtype),
                    "{dtype}, channel {channel}"
                );
            }
        }
        assert!(StepsWriter::new(&half_steps(), EncoderDtype::Uint8).is_none());
    }
}
