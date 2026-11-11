//! Splitting a run of interleaved RGB pixels into its three channel runs: the
//! inner step of the patch layouts that store a patch channel by channel. On
//! AArch64 the vector unit's de-interleaving load reads sixteen pixels into
//! their three planes in one instruction; on x86-64 three byte shuffles per
//! plane gather the same sixteen pixels; elsewhere a plain loop does the
//! same copies. Pure copies, so every path writes the same bytes.
//!
//! The AArch64 path is the module's only use of `core::arch` vector loads and
//! stores, hence the local override of the workspace-wide
//! `unsafe_code = "deny"`; every pointer it forms comes from an exact-size
//! chunk of a slice. The x86-64 path carries no `unsafe`: `multiversion`
//! selects its AVX2 clone at the first call, and every vector word enters and
//! leaves it through a `bytemuck` cast of an exact-size byte array.
#![cfg_attr(target_arch = "aarch64", allow(unsafe_code))]

/// The run splitter for this machine: the vector path where the vector unit
/// was detected, the scalar one otherwise. Only [`detect`](Self::detect)
/// builds one (and the test-only `scalar`), so the vector
/// call it makes rests on a detection of its own, not on a flag a caller
/// passes; a caller detects once and splits many runs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RgbSplitter {
    vector: bool,
}

impl RgbSplitter {
    /// The splitter for this machine, the detection done once.
    #[must_use]
    pub(crate) fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        let vector = std::arch::is_aarch64_feature_detected!("neon");
        #[cfg(target_arch = "x86_64")]
        let vector = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let vector = false;
        Self { vector }
    }

    /// The scalar path, for comparing the two.
    #[cfg(test)]
    pub(crate) fn scalar() -> Self {
        Self { vector: false }
    }

    /// `pixels` (`3 * n` bytes, interleaved) into `r`, `g` and `b` (`n` bytes
    /// each).
    #[inline]
    pub(crate) fn split(self, pixels: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
        debug_assert_eq!(pixels.len(), r.len() * 3);
        debug_assert_eq!(r.len(), g.len());
        debug_assert_eq!(r.len(), b.len());
        #[cfg(target_arch = "aarch64")]
        if self.vector {
            // SAFETY: `vector` is set by `detect` alone, which detected NEON.
            unsafe { neon::split(pixels, r, g, b) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if self.vector {
            avx2::split(pixels, r, g, b);
            return;
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = self.vector;
        split_scalar(pixels, r, g, b);
    }
}

fn split_scalar(pixels: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
    for (((pixel, r), g), b) in pixels
        .as_chunks::<3>()
        .0
        .iter()
        .zip(r.iter_mut())
        .zip(g.iter_mut())
        .zip(b.iter_mut())
    {
        *r = pixel[0];
        *g = pixel[1];
        *b = pixel[2];
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{vld3_u8, vld3q_u8, vst1_u8, vst1q_u8};

    /// Sixteen pixels per de-interleaving load, then eight, then the rest
    /// one by one; every load and store on an exact-size chunk.
    #[target_feature(enable = "neon")]
    pub(super) fn split(pixels: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
        let (wide, rest) = pixels.as_chunks::<48>();
        let (r_wide, r_rest) = r.as_chunks_mut::<16>();
        let (g_wide, g_rest) = g.as_chunks_mut::<16>();
        let (b_wide, b_rest) = b.as_chunks_mut::<16>();
        for (((pixels, r), g), b) in wide
            .iter()
            .zip(r_wide.iter_mut())
            .zip(g_wide.iter_mut())
            .zip(b_wide.iter_mut())
        {
            // SAFETY: `pixels` is 48 bytes (sixteen pixels); `r`, `g`, `b`
            // are sixteen bytes each.
            unsafe {
                let planes = vld3q_u8(pixels.as_ptr());
                vst1q_u8(r.as_mut_ptr(), planes.0);
                vst1q_u8(g.as_mut_ptr(), planes.1);
                vst1q_u8(b.as_mut_ptr(), planes.2);
            }
        }
        let (half, rest) = rest.as_chunks::<24>();
        let (r_half, r_rest) = r_rest.as_chunks_mut::<8>();
        let (g_half, g_rest) = g_rest.as_chunks_mut::<8>();
        let (b_half, b_rest) = b_rest.as_chunks_mut::<8>();
        for (((pixels, r), g), b) in half
            .iter()
            .zip(r_half.iter_mut())
            .zip(g_half.iter_mut())
            .zip(b_half.iter_mut())
        {
            // SAFETY: `pixels` is 24 bytes (eight pixels); `r`, `g`, `b` are
            // eight bytes each.
            unsafe {
                let planes = vld3_u8(pixels.as_ptr());
                vst1_u8(r.as_mut_ptr(), planes.0);
                vst1_u8(g.as_mut_ptr(), planes.1);
                vst1_u8(b.as_mut_ptr(), planes.2);
            }
        }
        super::split_scalar(rest, r_rest, g_rest, b_rest);
    }
}

/// The x86-64 path: a plane of sixteen pixels is three byte shuffles, one per
/// 16-byte word of the run, combined with two ors; eight pixels take one word
/// and a half. `multiversion` compiles the entry with AVX2 and without and
/// selects the clone at the first call; the clone without AVX2 is the plain
/// loop, reached only through a splitter built without the detection, which
/// none is. No `unsafe`: the words move through `bytemuck` casts of
/// exact-size arrays. The detection is the crate's for its other x86-64
/// paths; the shuffles themselves need SSSE3, which every AVX2 core has.
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use core::arch::x86_64::{__m128i, _mm_or_si128, _mm_shuffle_epi8};

    use multiversion::{multiversion, target::match_target};

    /// `pixels` (`3 * n` bytes, interleaved) into `r`, `g` and `b` (`n`
    /// bytes each).
    #[multiversion(targets("x86_64+avx2"))]
    pub(super) fn split(pixels: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
        match_target! {
            "x86_64+avx2" => split_avx2(pixels, r, g, b),
            _ => super::split_scalar(pixels, r, g, b),
        }
    }

    /// The shuffle-mask byte that takes source byte `src` of a run when it
    /// lies in 16-byte word `word` of the run, and zero (`0x80`) when it does
    /// not, so the or of one shuffle per word gathers the whole.
    pub(crate) const fn place(src: usize, word: usize) -> u8 {
        if src / 16 == word {
            (src % 16) as u8
        } else {
            0x80
        }
    }

    /// The masks that gather one plane of `pixels` pixels from the run's
    /// `WORDS` words: output byte `i` is source byte `3 i + channel`.
    /// Sixteen pixels span three words; eight pixels span a word and a half,
    /// the rest of the plane zero.
    const fn plane_masks<const WORDS: usize>(channel: usize, pixels: usize) -> [[u8; 16]; WORDS] {
        // The words hold the three channel bytes of every pixel.
        assert!(3 * pixels <= 16 * WORDS);
        let mut masks = [[0x80; 16]; WORDS];
        let mut word = 0;
        while word < WORDS {
            let mut i = 0;
            while i < pixels {
                masks[word][i] = place(3 * i + channel, word);
                i += 1;
            }
            word += 1;
        }
        masks
    }
    const RED16: [[u8; 16]; 3] = plane_masks(0, 16);
    const GREEN16: [[u8; 16]; 3] = plane_masks(1, 16);
    const BLUE16: [[u8; 16]; 3] = plane_masks(2, 16);
    const RED8: [[u8; 16]; 2] = plane_masks(0, 8);
    const GREEN8: [[u8; 16]; 2] = plane_masks(1, 8);
    const BLUE8: [[u8; 16]; 2] = plane_masks(2, 8);

    /// One plane of sixteen pixels from the run's three words.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn plane16(words: [__m128i; 3], masks: &[[u8; 16]; 3]) -> __m128i {
        _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(words[0], bytemuck::must_cast(masks[0])),
                _mm_shuffle_epi8(words[1], bytemuck::must_cast(masks[1])),
            ),
            _mm_shuffle_epi8(words[2], bytemuck::must_cast(masks[2])),
        )
    }

    /// Sixteen interleaved pixels (48 bytes) as their three planes of
    /// sixteen bytes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(in crate::vision) fn deinterleave16(run: [u8; 48]) -> [[u8; 16]; 3] {
        let words: [__m128i; 3] = bytemuck::must_cast(run);
        [
            bytemuck::must_cast(plane16(words, &RED16)),
            bytemuck::must_cast(plane16(words, &GREEN16)),
            bytemuck::must_cast(plane16(words, &BLUE16)),
        ]
    }

    /// The inverse: the masks that spread one `plane` over the three output
    /// words, where output word `k`'s byte `i` is byte `(16 k + i) / 3` of
    /// plane `(16 k + i) % 3`.
    const fn interleave_masks(plane: usize) -> [[u8; 16]; 3] {
        let mut masks = [[0x80; 16]; 3];
        let mut k = 0;
        while k < 3 {
            let mut i = 0;
            while i < 16 {
                let g = 16 * k + i;
                if g % 3 == plane {
                    masks[k][i] = (g / 3) as u8;
                }
                i += 1;
            }
            k += 1;
        }
        masks
    }
    const RED_OUT: [[u8; 16]; 3] = interleave_masks(0);
    const GREEN_OUT: [[u8; 16]; 3] = interleave_masks(1);
    const BLUE_OUT: [[u8; 16]; 3] = interleave_masks(2);

    /// Three planes of sixteen bytes as sixteen interleaved pixels (48
    /// bytes): the inverse of [`deinterleave16`].
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn interleave16(planes: [[u8; 16]; 3]) -> [u8; 48] {
        let [r, g, b]: [__m128i; 3] = bytemuck::must_cast(planes);
        let word = |k: usize| {
            _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(r, bytemuck::must_cast(RED_OUT[k])),
                    _mm_shuffle_epi8(g, bytemuck::must_cast(GREEN_OUT[k])),
                ),
                _mm_shuffle_epi8(b, bytemuck::must_cast(BLUE_OUT[k])),
            )
        };
        bytemuck::must_cast([word(0), word(1), word(2)])
    }

    /// One plane of eight pixels from the run's word and a half.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn plane8(words: [__m128i; 2], masks: &[[u8; 16]; 2]) -> [u8; 8] {
        let gathered = _mm_or_si128(
            _mm_shuffle_epi8(words[0], bytemuck::must_cast(masks[0])),
            _mm_shuffle_epi8(words[1], bytemuck::must_cast(masks[1])),
        );
        let halves: [[u8; 8]; 2] = bytemuck::must_cast(gathered);
        halves[0]
    }

    /// Sixteen pixels per step, then eight, then the rest one by one.
    #[target_feature(enable = "avx2")]
    fn split_avx2(pixels: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
        let (wide, rest) = pixels.as_chunks::<48>();
        let (r_wide, r_rest) = r.as_chunks_mut::<16>();
        let (g_wide, g_rest) = g.as_chunks_mut::<16>();
        let (b_wide, b_rest) = b.as_chunks_mut::<16>();
        for (((pixels, r), g), b) in wide
            .iter()
            .zip(r_wide.iter_mut())
            .zip(g_wide.iter_mut())
            .zip(b_wide.iter_mut())
        {
            [*r, *g, *b] = deinterleave16(*pixels);
        }
        let (half, rest) = rest.as_chunks::<24>();
        let (r_half, r_rest) = r_rest.as_chunks_mut::<8>();
        let (g_half, g_rest) = g_rest.as_chunks_mut::<8>();
        let (b_half, b_rest) = b_rest.as_chunks_mut::<8>();
        for (((pixels, r), g), b) in half
            .iter()
            .zip(r_half.iter_mut())
            .zip(g_half.iter_mut())
            .zip(b_half.iter_mut())
        {
            let thirds: [[u8; 8]; 3] = bytemuck::must_cast(*pixels);
            let words: [__m128i; 2] =
                bytemuck::must_cast([thirds[0], thirds[1], thirds[2], [0u8; 8]]);
            *r = plane8(words, &RED8);
            *g = plane8(words, &GREEN8);
            *b = plane8(words, &BLUE8);
        }
        super::split_scalar(rest, r_rest, g_rest, b_rest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs of every length from 0 to 40 pixels (whole sixteens, eights and
    /// remainders): the three channel runs are the pixels' bytes in order,
    /// on the vector path and the scalar one alike.
    #[test]
    fn every_run_length_splits_into_its_channels() {
        for n in 0..=40 {
            let pixels: Vec<u8> = (0..3 * n).map(|i| (i * 37 % 251) as u8).collect();
            for splitter in [RgbSplitter::detect(), RgbSplitter::scalar()] {
                let (mut r, mut g, mut b) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
                splitter.split(&pixels, &mut r, &mut g, &mut b);
                for i in 0..n {
                    assert_eq!(
                        (r[i], g[i], b[i]),
                        (pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2]),
                        "n {n} {splitter:?} pixel {i}"
                    );
                }
            }
        }
    }
}
