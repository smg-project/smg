//! Splitting a run of interleaved RGB pixels into its three channel runs: the
//! inner step of the patch layouts that store a patch channel by channel. On
//! AArch64 the vector unit's de-interleaving load reads sixteen pixels into
//! their three planes in one instruction; elsewhere a plain loop does the
//! same copies. Pure copies, so every path writes the same bytes.
//!
//! The AArch64 path is the module's only use of `core::arch` vector loads and
//! stores, hence the local override of the workspace-wide
//! `unsafe_code = "deny"`; every pointer it forms comes from an exact-size
//! chunk of a slice.
#![cfg_attr(target_arch = "aarch64", allow(unsafe_code))]

/// The run splitter for this machine: the vector path where the vector unit
/// was detected, the scalar one otherwise. Only [`detect`](Self::detect)
/// builds one (and the test-only [`scalar`](Self::scalar)), so the vector
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
        #[cfg(not(target_arch = "aarch64"))]
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
        #[cfg(not(target_arch = "aarch64"))]
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
