//! Runtime (dlopen) binding to libjpeg-turbo's TurboJPEG API for JPEG decode.
//!
//! PIL/Pillow (and therefore vLLM) decode JPEGs with libjpeg-turbo using its
//! default options: accurate (islow) integer IDCT and "fancy" (bilinear) chroma
//! upsampling. The pure-Rust `image`/`zune-jpeg` decoder differs by a few levels
//! per pixel, which the vision encoder amplifies into a large embedding shift,
//! making TokenSpeed's multimodal accuracy diverge from vLLM. Decoding through
//! libjpeg-turbo with the same defaults makes SMG's pixel values match vLLM's.
//!
//! We load libturbojpeg at RUNTIME via `dlopen` rather than linking it, so the
//! crate (and every consumer — including the Go/Python bindings and CI builds
//! that don't ship libturbojpeg) compiles on any platform with no build script
//! and no link-time dependency. Where the shared library is present (the serving
//! image), decode goes through it for PIL parity; where it's absent,
//! `decode_jpeg_rgb` returns `None` and the caller falls back to the pure-Rust
//! decoder. Default flags (0) select accurate DCT + fancy upsampling, matching
//! Pillow.
//!
//! CMYK and YCCK JPEGs (four components, usually with an Adobe APP14 marker)
//! are decoded as CMYK, which is the one output TurboJPEG offers for them, and
//! converted to RGB the way PIL converts them: see [`pil_cmyk_to_rgb`].
//!
//! This module is one of the crate's FFI surfaces (with `webp_native` and the
//! optional `opencv_buffer`), so it locally overrides the workspace-wide
//! `unsafe_code = "deny"` for the C bindings.
#![allow(unsafe_code)]

use std::{
    fmt,
    os::raw::{c_int, c_uchar, c_ulong, c_void},
    str::FromStr,
    sync::OnceLock,
};

use image::{DynamicImage, RgbImage};
use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};

type TjHandle = *mut c_void;
/// TurboJPEG pixel formats (`enum TJPF`).
const TJPF_RGB: c_int = 0;
const TJPF_CMYK: c_int = 11;
/// TurboJPEG JPEG colorspaces (`enum TJCS`) as `tjDecompressHeader3` reports
/// them; a CMYK or YCCK JPEG can only be decompressed to CMYK pixels.
const TJCS_CMYK: c_int = 3;
const TJCS_YCCK: c_int = 4;

type TjInitDecompress = unsafe extern "C" fn() -> TjHandle;
type TjDecompressHeader3 = unsafe extern "C" fn(
    TjHandle,
    *const c_uchar,
    c_ulong,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;
type TjDecompress2 = unsafe extern "C" fn(
    TjHandle,
    *const c_uchar,
    c_ulong,
    *mut c_uchar,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
type TjDestroy = unsafe extern "C" fn(TjHandle) -> c_int;
type TjGetErrorCode = unsafe extern "C" fn(TjHandle) -> c_int;
/// `tjGetErrorCode`'s reading of a call that returned -1 (`enum TJERR`): a
/// warning was raised during the call, and a fatal error may still have
/// followed it (`TurboJpeg::warned`), or `TJERR_FATAL` (1), a fatal error
/// with no warning before it.
const TJERR_WARNING: c_int = 0;

/// Resolved TurboJPEG entry points. Holds the loaded `Library` so the function
/// pointers stay valid for the process lifetime.
struct TurboJpeg {
    _lib: Library,
    init: TjInitDecompress,
    header: TjDecompressHeader3,
    decompress: TjDecompress2,
    destroy: TjDestroy,
    /// TurboJPEG 2.0 and later; a 1.x library has no such reading, and every
    /// non-zero return is a failure with it.
    error_code: Option<TjGetErrorCode>,
}

impl TurboJpeg {
    /// Whether the last call on `handle` raised a warning: the library notes
    /// a defect in the data (stray bytes before a marker, a corrupt scan, a
    /// scan ending early) as a warning, reads or writes past it and returns
    /// -1 at its end. The reading is a flag the call sets at its first
    /// warning and never clears, so a fatal error that follows a warning in
    /// the same call reads as a warning too: the caller checks the frame
    /// itself. A 1.x library, without the reading, warns of nothing.
    ///
    /// # Safety
    /// `handle` is a live handle of this library, and the call read is the
    /// last one made on it.
    unsafe fn warned(&self, handle: TjHandle) -> bool {
        self.error_code
            .is_some_and(|code| unsafe { code(handle) } == TJERR_WARNING)
    }
}

// The function pointers are plain C entry points with no shared mutable state;
// the library handle is kept alive for the process and never mutated.
unsafe impl Send for TurboJpeg {}
unsafe impl Sync for TurboJpeg {}

/// The names the library is tried under: the runtime soname first (shipped
/// by the runtime package), then the dev symlink and common macOS names.
const CANDIDATES: &[&str] = &[
    "libturbojpeg.so.0",
    "libturbojpeg.so",
    "libturbojpeg.0.dylib",
    "libturbojpeg.dylib",
];

fn load_turbojpeg() -> Option<TurboJpeg> {
    load_turbojpeg_from(CANDIDATES)
}

/// Load the library under the first of `candidates` the loader resolves, or
/// `None` when none does: the seam a test uses to stand in a loader that
/// finds nothing, without touching the process-wide binding.
fn load_turbojpeg_from(candidates: &[&str]) -> Option<TurboJpeg> {
    // SAFETY: loading a system shared library by name; we only resolve the five
    // documented TurboJPEG symbols below and keep the handle for their lifetime.
    let lib = candidates
        .iter()
        .find_map(|name| unsafe { Library::new(name) }.ok())?;
    // SAFETY: each symbol is resolved against the just-loaded library with the
    // signature documented by the TurboJPEG API. We copy the bare function
    // pointers out (dropping the borrowing `Symbol`s) and keep `lib` alive in
    // the returned struct, so the pointers remain valid.
    let (init, header, decompress, destroy, error_code) = unsafe {
        let init: Symbol<TjInitDecompress> = lib.get(b"tjInitDecompress\0").ok()?;
        let header: Symbol<TjDecompressHeader3> = lib.get(b"tjDecompressHeader3\0").ok()?;
        let decompress: Symbol<TjDecompress2> = lib.get(b"tjDecompress2\0").ok()?;
        let destroy: Symbol<TjDestroy> = lib.get(b"tjDestroy\0").ok()?;
        let error_code: Option<Symbol<TjGetErrorCode>> = lib.get(b"tjGetErrorCode\0").ok();
        (
            *init,
            *header,
            *decompress,
            *destroy,
            error_code.map(|symbol| *symbol),
        )
    };
    Some(TurboJpeg {
        _lib: lib,
        init,
        header,
        decompress,
        destroy,
        error_code,
    })
}

/// Process-wide cached TurboJPEG binding, or `None` if the library is absent.
fn turbojpeg() -> Option<&'static TurboJpeg> {
    static TJ: OnceLock<Option<TurboJpeg>> = OnceLock::new();
    TJ.get_or_init(load_turbojpeg).as_ref()
}

/// Cap on the decoded buffer (RGB, or CMYK for a four-component JPEG). Matches
/// the `image` crate's default 512 MiB limit, which `decode_image_bytes`
/// reserves against before the pure-Rust fallback decodes (on the RGB size, so
/// a CMYK JPEG refused here for its 4-byte buffer may still decode there).
const MAX_DECODED_BYTES: usize = 512 * 1024 * 1024;

/// Byte length of a `w` x `h` buffer of `channels` bytes per pixel, or `None`
/// if it overflows or exceeds `MAX_DECODED_BYTES`.
fn decoded_buffer_len(w: usize, h: usize, channels: usize) -> Option<usize> {
    w.checked_mul(h)?
        .checked_mul(channels)
        .filter(|&n| n <= MAX_DECODED_BYTES)
}

/// One CMYK pixel as libjpeg hands it out (a YCCK source already converted to
/// CMYK), to the RGB PIL produces for it.
///
/// PIL reads every four-component JPEG with the raw mode `CMYK;I` ("assume
/// Adobe conventions"): the four bytes are inverted, whether or not the file
/// carries an Adobe APP14 marker. `convert("RGB")` then applies `cmyk2rgb`
/// from libImaging/Convert.c to the inverted values `C' = 255 - C`, ...,
/// `K' = 255 - K`: with `nk = 255 - K'`, `R = CLIP8(nk - MULDIV255(C', nk))`,
/// the same for G from M' and B from Y', where `MULDIV255(a, b)` is
/// `(t + (t >> 8)) >> 8` for `t = a * b + 128` (a rounded `a * b / 255`).
///
/// On the bytes libjpeg hands out that is `R = MULDIV255(C, K)`: `nk` is the
/// raw `K`, and `K - MULDIV255(255 - C, K)` equals `MULDIV255(C, K)` for every
/// pair of bytes (a test checks all 65,536), so the clip never engages.
#[inline]
pub fn pil_cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> [u8; 3] {
    let k = u32::from(k);
    let channel = |ink: u8| {
        let t = u32::from(ink) * k + 128;
        // At most (255 * 255 + 128 + 254) >> 8 = 255: fits.
        ((t + (t >> 8)) >> 8) as u8
    };
    [channel(c), channel(m), channel(y)]
}

/// Convert a packed CMYK8 buffer (as decoded by libjpeg) to packed RGB8, PIL's
/// way, in place: pixel `i` is read from `4i..4i+4` before `3i..3i+3` is
/// written, so no unread byte is overwritten and the decoded buffer stays the
/// one allocation `MAX_DECODED_BYTES` bounds. On AArch64 and x86-64 the
/// vector unit does sixteen pixels at a time with the same arithmetic in
/// 16-bit lanes; the rest, and every other target, is the formula pixel by
/// pixel.
fn cmyk_to_rgb_like_pil(mut buf: Vec<u8>) -> Vec<u8> {
    let pixels = buf.len() / 4;
    #[cfg(target_arch = "aarch64")]
    let start = neon::convert_in_place(&mut buf[..4 * pixels]);
    #[cfg(target_arch = "x86_64")]
    let start = avx2::convert_in_place(&mut buf[..4 * pixels]);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let start = 0;
    convert_pixels(&mut buf[..4 * pixels], start);
    buf.truncate(pixels * 3);
    buf
}

/// Pixels `from` onwards of `buf` (`4 * n` bytes of CMYK) converted in place
/// to RGB at `3i` by the pixel formula; returns `n`.
fn convert_pixels(buf: &mut [u8], from: usize) -> usize {
    let pixels = buf.len() / 4;
    for i in from..pixels {
        let [c, m, y, k] = [buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]];
        buf[3 * i..3 * i + 3].copy_from_slice(&pil_cmyk_to_rgb(c, m, y, k));
    }
    pixels
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        uint16x8_t, uint8x16_t, uint8x16x3_t, vaddq_u16, vcombine_u8, vdupq_n_u16, vget_low_u8,
        vld4q_u8, vmlal_high_u8, vmlal_u8, vshrn_n_u16, vshrq_n_u16, vst3q_u8,
    };

    /// The leading whole blocks of sixteen CMYK pixels of `buf` (`4 * n`
    /// bytes) converted in place to RGB at `3i`; returns the pixels done, the
    /// rest (fewer than sixteen) is the caller's. Zero on a core without the
    /// vector unit.
    pub(super) fn convert_in_place(buf: &mut [u8]) -> usize {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return 0;
        }
        // SAFETY: the vector unit was detected just above.
        unsafe { convert_blocks(buf) }
    }

    /// `MULDIV255(ink, k)` on sixteen pixels in 16-bit lanes: `ink * k + 128`
    /// is at most 65,153 and `t + (t >> 8)` at most 65,407, both within a
    /// `u16`, and the high byte of the latter is the scalar formula's value.
    #[inline]
    #[target_feature(enable = "neon")]
    fn muldiv255(ink: uint8x16_t, k: uint8x16_t) -> uint8x16_t {
        let bias: uint16x8_t = vdupq_n_u16(128);
        let low = vmlal_u8(bias, vget_low_u8(ink), vget_low_u8(k));
        let high = vmlal_high_u8(bias, ink, k);
        let low = vaddq_u16(low, vshrq_n_u16::<8>(low));
        let high = vaddq_u16(high, vshrq_n_u16::<8>(high));
        vcombine_u8(vshrn_n_u16::<8>(low), vshrn_n_u16::<8>(high))
    }

    #[target_feature(enable = "neon")]
    fn convert_blocks(buf: &mut [u8]) -> usize {
        let pixels = buf.len() / 4;
        let full = pixels - pixels % 16;
        let base = buf.as_mut_ptr();
        let mut i = 0;
        while i < full {
            // SAFETY: `i + 16 <= full <= pixels`, so the 64 bytes read at
            // `4 * i` and the 48 bytes written at `3 * i` (which end before
            // `4 * i + 64`) lie within `buf`; the read completes before the
            // write, so their overlap within this block is harmless, and no
            // pixel still to be read is overwritten.
            unsafe {
                let pixel = vld4q_u8(base.add(4 * i).cast_const());
                let rgb = uint8x16x3_t(
                    muldiv255(pixel.0, pixel.3),
                    muldiv255(pixel.1, pixel.3),
                    muldiv255(pixel.2, pixel.3),
                );
                vst3q_u8(base.add(3 * i), rgb);
            }
            i += 16;
        }
        full
    }
}

/// The x86-64 path of the CMYK conversion: sixteen pixels per step, the four
/// planes gathered with byte shuffles, `MULDIV255` in 16-bit lanes, the three
/// result planes rejoined with the channel-run module's shuffles. The entry
/// is dispatched by `multiversion`; the clone for a core without AVX2 is the
/// pixel formula. No `unsafe` of its own: the words move through `bytemuck`
/// casts of exact-size arrays.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_mullo_epi16, _mm_or_si128, _mm_packus_epi16, _mm_set1_epi16,
        _mm_setzero_si128, _mm_shuffle_epi8, _mm_srli_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };

    use multiversion::{multiversion, target::match_target};

    use crate::vision::rgb_runs::avx2::{interleave16, place};

    /// The leading whole blocks of sixteen CMYK pixels of `buf` (`4 * n`
    /// bytes) converted in place to RGB at `3i`; returns the pixels done, the
    /// rest (fewer than sixteen) is the caller's. Zero on a core without the
    /// vector unit.
    pub(super) fn convert_in_place(buf: &mut [u8]) -> usize {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return 0;
        }
        convert_blocks(buf)
    }

    #[multiversion(targets("x86_64+avx2"))]
    fn convert_blocks(buf: &mut [u8]) -> usize {
        match_target! {
            "x86_64+avx2" => convert_blocks_avx2(buf),
            _ => super::convert_pixels(buf, 0),
        }
    }

    /// The masks that gather plane `p` of sixteen four-byte pixels from the
    /// block's four words: byte `i` is source byte `4 i + p`, in word
    /// `i / 4` of the four.
    const fn plane_masks(p: usize) -> [[u8; 16]; 4] {
        let mut masks = [[0x80; 16]; 4];
        let mut word = 0;
        while word < 4 {
            let mut i = 0;
            while i < 16 {
                masks[word][i] = place(4 * i + p, word);
                i += 1;
            }
            word += 1;
        }
        masks
    }
    const PLANE_C: [[u8; 16]; 4] = plane_masks(0);
    const PLANE_M: [[u8; 16]; 4] = plane_masks(1);
    const PLANE_Y: [[u8; 16]; 4] = plane_masks(2);
    const PLANE_K: [[u8; 16]; 4] = plane_masks(3);

    /// One plane of sixteen pixels from the block's four words.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn plane(words: [__m128i; 4], masks: &[[u8; 16]; 4]) -> __m128i {
        _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(words[0], bytemuck::must_cast(masks[0])),
                _mm_shuffle_epi8(words[1], bytemuck::must_cast(masks[1])),
            ),
            _mm_or_si128(
                _mm_shuffle_epi8(words[2], bytemuck::must_cast(masks[2])),
                _mm_shuffle_epi8(words[3], bytemuck::must_cast(masks[3])),
            ),
        )
    }

    /// `MULDIV255(ink, k)` on sixteen pixels in 16-bit lanes: `ink * k + 128`
    /// is at most 65,153 and `t + (t >> 8)` at most 65,407, both within a
    /// `u16`, and the high byte of the latter is the scalar formula's value.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn muldiv255(ink: __m128i, k: __m128i) -> __m128i {
        let zero = _mm_setzero_si128();
        let bias = _mm_set1_epi16(128);
        let low = _mm_add_epi16(
            _mm_mullo_epi16(_mm_unpacklo_epi8(ink, zero), _mm_unpacklo_epi8(k, zero)),
            bias,
        );
        let high = _mm_add_epi16(
            _mm_mullo_epi16(_mm_unpackhi_epi8(ink, zero), _mm_unpackhi_epi8(k, zero)),
            bias,
        );
        let low = _mm_add_epi16(low, _mm_srli_epi16::<8>(low));
        let high = _mm_add_epi16(high, _mm_srli_epi16::<8>(high));
        _mm_packus_epi16(_mm_srli_epi16::<8>(low), _mm_srli_epi16::<8>(high))
    }

    /// The whole blocks of sixteen pixels: the 64 bytes at `4 i` are read
    /// before the 48 bytes at `3 i` are written (which end before `4 i + 64`),
    /// so no pixel still to be read is overwritten.
    #[target_feature(enable = "avx2")]
    fn convert_blocks_avx2(buf: &mut [u8]) -> usize {
        let pixels = buf.len() / 4;
        let full = pixels - pixels % 16;
        let mut i = 0;
        while i < full {
            let block: [u8; 64] = bytemuck::pod_read_unaligned(&buf[4 * i..4 * i + 64]);
            let words: [__m128i; 4] = bytemuck::must_cast(block);
            let k = plane(words, &PLANE_K);
            let rgb = interleave16([
                bytemuck::must_cast(muldiv255(plane(words, &PLANE_C), k)),
                bytemuck::must_cast(muldiv255(plane(words, &PLANE_M), k)),
                bytemuck::must_cast(muldiv255(plane(words, &PLANE_Y), k)),
            ]);
            buf[3 * i..3 * i + 48].copy_from_slice(&rgb);
            i += 16;
        }
        full
    }
}

/// Whether libjpeg-turbo was found, so JPEGs decode to PIL's pixels.
pub fn turbojpeg_available() -> bool {
    turbojpeg().is_some()
}

/// What a process does when the library cannot be loaded.
///
/// `Prefer`, the default, decodes JPEGs with the pure-Rust decoder then,
/// whose pixels differ from the reference decoder's by a few levels on a
/// share of every JPEG's bytes; `Require` refuses to come up without the
/// library, for a deployment whose pixels must be the reference decoder's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurboJpegPolicy {
    #[default]
    Prefer,
    Require,
}

impl TurboJpegPolicy {
    /// Parse from a case-insensitive string (`prefer` | `require`).
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "prefer" => Some(Self::Prefer),
            "require" => Some(Self::Require),
            _ => None,
        }
    }

    /// Canonical lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefer => "prefer",
            Self::Require => "require",
        }
    }
}

impl fmt::Display for TurboJpegPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TurboJpegPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
            .ok_or_else(|| format!("invalid TurboJPEG policy '{s}'; expected prefer|require"))
    }
}

/// The library could not be loaded and the policy is
/// [`TurboJpegPolicy::Require`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "libturbojpeg.so.0 (libjpeg-turbo's TurboJPEG library) could not be loaded and the TurboJPEG \
     policy is `require`: without it JPEGs decode with the pure-Rust decoder, whose pixels differ \
     from the reference decoder's by a few levels; install the libjpeg-turbo runtime package where \
     this process runs, or set the policy to `prefer` to serve with the fallback decoder"
)]
pub struct TurboJpegUnavailable;

/// Apply `policy` to this process: `Ok` unless the policy requires the
/// library and the loader did not find it.
pub fn enforce_policy(policy: TurboJpegPolicy) -> Result<(), TurboJpegUnavailable> {
    enforce(policy, turbojpeg_available())
}

/// The decision for an availability the caller states: the process-wide
/// loader's in [`enforce_policy`], a stated one in tests.
fn enforce(policy: TurboJpegPolicy, available: bool) -> Result<(), TurboJpegUnavailable> {
    match policy {
        TurboJpegPolicy::Require if !available => Err(TurboJpegUnavailable),
        TurboJpegPolicy::Prefer | TurboJpegPolicy::Require => Ok(()),
    }
}

/// True if `bytes` start with the JPEG SOI marker.
pub fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

/// What a JPEG's frame header says: the dimensions and the component count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameHeader {
    width: i32,
    height: i32,
    components: u8,
}

/// The first marker segment of `bytes` whose marker `wanted` accepts: the
/// marker, its payload and the offset of what follows the segment. Fill bytes,
/// stray bytes before a marker and a stuffed zero (`FF 00`; defects both
/// decoders warn of and read past, one byte at a time) and the standalone
/// markers (the restart markers, SOI and TEM) are stepped over. The walk ends
/// at the end of the image, at a segment cut short, and at the scan header
/// unless that is the segment wanted, since no marker walk crosses the
/// entropy-coded data behind it.
fn segment(bytes: &[u8], wanted: impl Fn(u8) -> bool) -> Option<(u8, &[u8], usize)> {
    if !is_jpeg(bytes) {
        return None;
    }
    let mut at = 2;
    loop {
        // Stray bytes before a marker are stepped over to the next 0xFF, as
        // libjpeg's marker reader does.
        at += bytes.get(at..)?.iter().position(|&byte| byte == 0xFF)?;
        let marker = *bytes.get(at + 1)?;
        match marker {
            // A fill byte before the next marker.
            0xFF => at += 1,
            // A stuffed zero: no marker. libjpeg discards the pair and scans
            // on from the next byte, so a marker inside what a length behind
            // the pair would span is the one it reads; the walk does the same.
            0x00 => at += 2,
            // Standalone markers: the restart markers, SOI and TEM.
            0xD0..=0xD8 | 0x01 => at += 2,
            // The end of the image.
            0xD9 => return None,
            _ => {
                let length = usize::from(u16::from_be_bytes([
                    *bytes.get(at + 2)?,
                    *bytes.get(at + 3)?,
                ]));
                // The length counts its own two bytes.
                let end = at + 2 + length;
                let payload = bytes.get(at + 4..end)?;
                if wanted(marker) {
                    return Some((marker, payload, end));
                }
                if marker == 0xDA {
                    return None;
                }
                at = end;
            }
        }
    }
}

/// Whether `marker` opens a frame header: SOF0 to SOF15, less DHT, JPG and
/// DAC, which share the range.
fn is_frame_header(marker: u8) -> bool {
    matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC)
}

/// The frame header of a JPEG, read from its marker segments. The TurboJPEG
/// header call reports the same when it can classify the frame's sampling;
/// libjpeg-turbo 2.x refuses a frame whose components are sampled unlike each
/// other (a CMYK file with subsampled inks and a full-resolution black) with
/// "Could not determine subsampling type", though its decoder takes such a
/// frame, so this reader supplies the component count the call did not, for
/// a frame whose dimensions agree with the call's (`fallback_frame`). `None`
/// when the scan or the end of the image comes before a frame header, or a
/// segment is cut short.
fn frame_header(bytes: &[u8]) -> Option<FrameHeader> {
    let (_, payload, _) = segment(bytes, is_frame_header)?;
    let (&[_precision, h0, h1, w0, w1, components], _) = payload.split_first_chunk::<6>()?;
    let (height, width) = (u16::from_be_bytes([h0, h1]), u16::from_be_bytes([w0, w1]));
    (width > 0 && height > 0).then(|| FrameHeader {
        width: i32::from(width),
        height: i32::from(height),
        components,
    })
}

/// Where a JPEG's data ends, read behind the scan header as libjpeg reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// At the end-of-image marker.
    Marker,
    /// Behind a segment libjpeg reads past, with no marker after it: the
    /// entropy-coded data ended at that segment's marker.
    Segment,
    /// With no marker behind the scan header: in the entropy-coded data, or
    /// in bytes behind it that hold no marker.
    Data,
    /// At a marker libjpeg refuses behind a scan, at a segment with a length
    /// libjpeg refuses, or with no scan header.
    Refused,
}

/// Where a JPEG's data ends and what the walk to it stepped over: a scan
/// header, which libjpeg refuses behind a single-scan frame's scan, and a
/// table segment (DHT, DAC, DQT, DRI), which libjpeg parses there and
/// refuses when malformed, where the walk only steps over it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScanEnd {
    end: End,
    scan_header: bool,
    table: bool,
}

/// Where a JPEG's data ends, read behind the scan header as libjpeg reads it:
/// a stuffed byte (`FF 00`), fill bytes, restart markers and TEM are data; a
/// segment behind a scan (a table, the restart interval, the line count, an
/// application segment, a comment, the next scan's header) is stepped over
/// by its length, so a marker inside its payload is not the image's, and
/// the data may end inside it; the scan header and the tables among them
/// are noted, since libjpeg parses those where it skips the rest; a marker
/// libjpeg refuses there (a second start of image, a frame header, a
/// reserved marker) fails the file, so no marker behind it is the end
/// either. Bytes after the end marker, a common appendage, do not matter; a
/// marker inside a segment before the scan (an embedded thumbnail's) does
/// not count. One pass over the data, no allocation.
fn scan_end(bytes: &[u8]) -> ScanEnd {
    let mut tail = ScanEnd {
        end: End::Refused,
        scan_header: false,
        table: false,
    };
    let Some((_, _, mut at)) = segment(bytes, |marker| marker == 0xDA) else {
        return tail;
    };
    tail.end = End::Data;
    loop {
        let Some(next) = bytes
            .get(at..)
            .and_then(|rest| rest.iter().position(|&byte| byte == 0xFF))
        else {
            return tail;
        };
        at += next;
        let Some(&marker) = bytes.get(at + 1) else {
            return tail;
        };
        match marker {
            // A fill byte before a marker.
            0xFF => at += 1,
            // A stuffed zero, TEM or a restart marker: the decoder's data.
            0x00 | 0x01 | 0xD0..=0xD7 => at += 2,
            0xD9 => {
                tail.end = End::Marker;
                return tail;
            }
            // The segments libjpeg reads behind a scan: the Huffman and
            // arithmetic tables (DHT, DAC), the quantization tables (DQT),
            // the restart interval (DRI), the line count (DNL), the next
            // scan's header (SOS), application segments and comments. The
            // length counts its own two bytes.
            0xC4 | 0xCC | 0xDA..=0xDD | 0xE0..=0xEF | 0xFE => {
                tail.end = End::Segment;
                let (Some(&high), Some(&low)) = (bytes.get(at + 2), bytes.get(at + 3)) else {
                    return tail;
                };
                let length = usize::from(u16::from_be_bytes([high, low]));
                match marker {
                    // The next scan's header: libjpeg parses it and refuses a
                    // length short of its own bytes; whether a second scan is
                    // refused depends on the frame, the caller's to read.
                    0xDA => {
                        if length < 2 {
                            tail.end = End::Refused;
                            return tail;
                        }
                        tail.scan_header = true;
                    }
                    // A table or the restart interval: libjpeg parses it and
                    // refuses a length short of its own bytes (the restart
                    // interval's is four).
                    0xC4 | 0xCC | 0xDB | 0xDD => {
                        if length < 2 || (marker == 0xDD && length != 4) {
                            tail.end = End::Refused;
                            return tail;
                        }
                        tail.table = true;
                    }
                    // An application segment, a comment or the line count:
                    // skipped by the length past its own two bytes, nothing
                    // for a shorter one.
                    _ => {}
                }
                at += 2 + length.max(2);
            }
            // A second start of image, a frame header or a reserved
            // marker: libjpeg fails the file.
            _ => {
                tail.end = End::Refused;
                return tail;
            }
        }
    }
}

/// Whether a JPEG is decoded in one scan: a sequential frame (not a
/// progressive one) whose first scan names every component. libjpeg reads
/// such a file row by row and the reference decoder serves the rows as they
/// come; a progressive or multi-scan file is read through to its end marker
/// before the first row.
fn single_scan(bytes: &[u8]) -> bool {
    let Some((marker, frame, _)) = segment(bytes, is_frame_header) else {
        return false;
    };
    // SOF2, SOF6, SOF10 and SOF14 are the progressive frames.
    if matches!(marker, 0xC2 | 0xC6 | 0xCA | 0xCE) {
        return false;
    }
    let Some((_, scan, _)) = segment(bytes, |marker| marker == 0xDA) else {
        return false;
    };
    // The frame's component count is the sixth byte of its header, the
    // scan's the first of its own.
    frame
        .get(5)
        .is_some_and(|count| scan.first() == Some(count))
}

/// Whether the reference decoder serves a frame from a file that ends as
/// this one does, read without a decode. The decoder ignores a missing
/// end-of-image marker once every row is out, and a single-scan file's rows
/// are out once a marker ends its entropy-coded data; it refuses a file
/// whose entropy-coded data runs out (a scan cut short, or a bare end with
/// nothing behind it: it asks for bytes past the end and finds none), a
/// marker libjpeg refuses, and a progressive or multi-scan file that does
/// not reach its end marker, since it reads those through to the marker
/// before the first row. A second scan header behind a single-scan frame's
/// scan is refused by libjpeg when it reaches it, end marker or not. Behind
/// such a scan libjpeg parses a table segment and refuses a malformed one,
/// which the walk does not tell, so a table there is refused, end marker or
/// not. Three files the reference serves are refused here: a single-scan
/// file whose entropy-coded data is followed by bytes holding no marker
/// (the walk cannot tell them from the data, and whether the decoder's read
/// past the data was satisfied is the decoder's to know); one whose data is
/// followed by a valid table, with or without the end marker behind it (a
/// layout no encoder writes: the table serves no later scan); and one with
/// a restart interval whose scan data holds a reserved marker, which
/// libjpeg resyncs past at the next restart boundary where the walk reads
/// a marker it refuses.
pub(crate) fn end_is_served(bytes: &[u8]) -> bool {
    let tail = scan_end(bytes);
    match tail.end {
        End::Marker => !((tail.scan_header || tail.table) && single_scan(bytes)),
        End::Segment => !tail.scan_header && !tail.table && single_scan(bytes),
        End::Data | End::Refused => false,
    }
}

/// The pattern a frame's last row carries into the library's decode and is
/// read for after a -1 with a warning: the library writes the frame top to
/// bottom, so a last row still holding it means a fatal error ended the write
/// short of the frame's end. Two bytes alternate, so a buffer the pool handed
/// over filled with one byte never reads as written; a real last row that
/// happens to match falls back to the pure-Rust decoder, the safe side.
const SENTINEL_ROW: [u8; 2] = [0x5A, 0xA5];

fn write_sentinel(row: &mut [u8]) {
    for (byte, pattern) in row.iter_mut().zip(SENTINEL_ROW.iter().cycle()) {
        *byte = *pattern;
    }
}

fn holds_sentinel(row: &[u8]) -> bool {
    row.iter()
        .zip(SENTINEL_ROW.iter().cycle())
        .all(|(byte, pattern)| byte == pattern)
}

/// The frame taken when the header call read the frame and refused only its
/// sampling: the library's own dimensions, filled before the refusal, and
/// the component count from the marker reader, whose frame must be the one
/// the library read (the dimensions agree). A file whose markers read two
/// ways would size the buffer by a frame the library never writes, and the
/// sentinel in the decode would prove nothing. Four components decode as
/// CMYK and three as RGB; the decoder has no colour conversion for any other
/// count.
fn fallback_frame(w: c_int, h: c_int, frame: Option<FrameHeader>) -> Option<(c_int, c_int, bool)> {
    let frame = frame?;
    (matches!(frame.components, 3 | 4) && (frame.width, frame.height) == (w, h)).then_some((
        w,
        h,
        frame.components == 4,
    ))
}

/// Decode a JPEG to an RGB8 `DynamicImage` via libjpeg-turbo (PIL-compatible
/// defaults). A CMYK or YCCK JPEG is decompressed to CMYK, the only output
/// TurboJPEG offers for it, and converted with [`pil_cmyk_to_rgb`]. A frame
/// the library finished with a warning (a defect in the data it decoded past)
/// is returned as the reference decoder returns it, unless the file ends
/// where that decoder refuses it (`end_is_served`). Returns `None` on
/// any failure — including libturbojpeg not being available at runtime — so
/// the caller can fall back to the pure-Rust decoder.
pub fn decode_jpeg_rgb(bytes: &[u8]) -> Option<DynamicImage> {
    if !is_jpeg(bytes) {
        return None;
    }
    let tj = turbojpeg()?;
    // SAFETY:
    // - `tj.init` returns a handle that is null-checked before use; every early
    //   return below calls `tj.destroy(handle)` first, so the handle is freed
    //   exactly once and never used after destruction.
    // - `bytes` is a live `&[u8]`; its ptr/len describe a valid immutable region
    //   for the duration of the calls (libjpeg-turbo only reads it).
    // - `buf` is an owned `Vec<u8>` whose length is exactly `w*h*channels`
    //   (overflow-checked) from the dimensions the header call read (when it
    //   refuses only the sampling, the frame header adds the component count
    //   and must name the same dimensions, `fallback_frame`; the decoder
    //   never writes past the size it is given), with `channels` the
    //   pixel size of the format requested (3 for `TJPF_RGB`, 4 for
    //   `TJPF_CMYK`); a pooled buffer has a larger capacity, the length is
    //   what matters. It is the sole alias passed to the decoder; with
    //   `pitch=0` (=> `w*channels`) the decoder writes at most `w*h*channels`
    //   bytes, so no out-of-bounds write occurs. The image is built only after
    //   the write is known complete, which covers every byte of the buffer
    //   (its contents before the write are a previous frame's, not zeros):
    //   `rc == 0`, or -1 with the library's warning reading and the frame's
    //   last row overwritten. The reading alone proves nothing: a fatal error
    //   after a warning in the same call keeps it, and leaves by a long jump
    //   before the write reaches the end of the frame, so the sentinel
    //   written into the last row before the call stays there.
    unsafe {
        let handle = (tj.init)();
        if handle.is_null() {
            return None;
        }
        let (mut w, mut h, mut subsamp, mut colorspace) = (0_i32, 0_i32, 0_i32, 0_i32);
        let hdr = (tj.header)(
            handle,
            bytes.as_ptr(),
            bytes.len() as c_ulong,
            &mut w,
            &mut h,
            &mut subsamp,
            &mut colorspace,
        );
        // A header the call read in full but warned about (stray bytes before
        // a marker, most often) has its dimensions, sampling and colorspace
        // filled: the decode goes on to meet the same defect and keeps its
        // frame, as the reference decoder does. A header the call refused
        // after a warning reads as warned too (the flag stays set), and is
        // told apart by the sampling it left unclassified: such a frame takes
        // the arms below. The reading comes before any other call on the
        // handle.
        let warned_header = hdr != 0 && subsamp >= 0 && tj.warned(handle);
        // The header call fills the dimensions once it has read the frame,
        // before it classifies the sampling; a header it could not read
        // leaves them at zero (3.x reports -1), and the file falls back as
        // before, with no buffer taken and no second read on the handle.
        if w <= 0 || h <= 0 {
            (tj.destroy)(handle);
            return None;
        }
        // A CMYK or YCCK source can only come out as CMYK; everything else
        // (YCbCr, grayscale, RGB) comes out as RGB. When the call read the
        // frame and refused only its sampling (libjpeg-turbo 2.x on
        // components sampled unlike each other), the frame header supplies
        // the component count (its frame the one the call read: the
        // dimensions agree), and four components mean CMYK: the decoder
        // takes the frame the call would not classify. The call writes that
        // refusal into the sampling before it throws. A frame of two or of
        // five and more components is refused on its sampling too, and the
        // decoder has no colour conversion for it: three and four components
        // alone take the fallback. Any other failure of the call is fatal,
        // and the file falls back.
        let (w, h, cmyk) = if hdr == 0 || warned_header {
            (w, h, colorspace == TJCS_CMYK || colorspace == TJCS_YCCK)
        } else if subsamp < 0 {
            match fallback_frame(w, h, frame_header(bytes)) {
                Some(frame) => frame,
                None => {
                    (tj.destroy)(handle);
                    return None;
                }
            }
        } else {
            (tj.destroy)(handle);
            return None;
        };
        let (wu, hu) = (w as usize, h as usize);
        let (pixel_format, channels) = if cmyk { (TJPF_CMYK, 4) } else { (TJPF_RGB, 3) };
        // Guard against absurd dimensions before allocating.
        let nbytes = match decoded_buffer_len(wu, hu, channels) {
            Some(n) => n,
            None => {
                (tj.destroy)(handle);
                return None;
            }
        };
        let mut buf = crate::buffer_pool::take(nbytes);
        // The frame's last row carries a sentinel into the call: the library
        // writes the frame top to bottom, so the row is overwritten only when
        // the write reached the end of the frame (`kept` below).
        let last_row = nbytes - wu * channels;
        write_sentinel(&mut buf[last_row..]);
        let rc = (tj.decompress)(
            handle,
            bytes.as_ptr(),
            bytes.len() as c_ulong,
            buf.as_mut_ptr(),
            w,
            0, // pitch = 0 -> width * pixelsize
            h,
            pixel_format,
            0, // default flags: accurate IDCT + fancy upsampling (matches Pillow)
        );
        // A -1 after the whole frame is written is a warning: the library
        // noted a defect in the data (stray bytes before a marker, a corrupt
        // scan) and went on, and the reference decoder serves the same pixels
        // from the same library, so the frame is kept when the file ends
        // where that decoder serves it (`end_is_served`) and the last row was
        // written. Not kept: a file cut short (the library fills the rest of
        // the frame and warns of the premature end, where the reference
        // decoder refuses the file), a fatal error, which reads as a warning
        // when a warning came first and leaves the sentinel in the last row,
        // and a library without the reading; the buffer goes back, so the
        // next request's take is warm, and the file falls back.
        let kept = rc == 0
            || (tj.warned(handle) && end_is_served(bytes) && !holds_sentinel(&buf[last_row..]));
        (tj.destroy)(handle);
        if !kept {
            crate::buffer_pool::recycle(buf);
            return None;
        }
        let rgb = if cmyk { cmyk_to_rgb_like_pil(buf) } else { buf };
        RgbImage::from_raw(w as u32, h as u32, rgb).map(DynamicImage::ImageRgb8)
    }
}

/// Whether the library found reads a call's warning (`tjGetErrorCode`,
/// TurboJPEG 2.0 and later), so a frame it decoded with a warning is kept:
/// a 1.x library falls back on every -1 and keeps none.
#[cfg(test)]
pub(crate) fn turbojpeg_reads_warnings() -> bool {
    turbojpeg().is_some_and(|tj| tj.error_code.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_buffer_len_caps_decoded_size() {
        assert_eq!(decoded_buffer_len(1024, 768, 3), Some(1024 * 768 * 3));
        assert_eq!(decoded_buffer_len(1024, 768, 4), Some(1024 * 768 * 4));
        // 65500 x 65500 is the largest size libjpeg-turbo accepts: ~12 GiB of RGB.
        assert_eq!(decoded_buffer_len(65500, 65500, 3), None);
        assert_eq!(decoded_buffer_len(usize::MAX, 2, 3), None);
    }

    /// Hand-computed from PIL's arithmetic (`CMYK;I` inversion, then
    /// `cmyk2rgb`): with the inverted bytes `(C, M, Y, K)` and `nk = 255 - K`,
    /// each channel is `nk - MULDIV255(ink, nk)`, `MULDIV255(a, b) =
    /// (t + (t >> 8)) >> 8`, `t = a * b + 128`.
    #[test]
    fn pil_cmyk_to_rgb_matches_pils_arithmetic() {
        // Raw 0 everywhere = full ink everywhere after the inversion: black.
        assert_eq!(pil_cmyk_to_rgb(0, 0, 0, 0), [0, 0, 0]);
        // Raw 255 everywhere = no ink: white.
        assert_eq!(pil_cmyk_to_rgb(255, 255, 255, 255), [255, 255, 255]);
        // Full magenta and yellow ink, no cyan, no black: red.
        assert_eq!(pil_cmyk_to_rgb(255, 0, 0, 255), [255, 0, 0]);
        // Any black ink at full strength (raw K = 0) wins: nk = 0.
        assert_eq!(pil_cmyk_to_rgb(245, 243, 18, 0), [0, 0, 0]);
        // Raw (29, 200, 60, 180) inverts to (226, 55, 195, 75), nk = 180:
        //   R: t = 226*180+128 = 40808, (40808 + 159) >> 8 = 160, 180-160 = 20
        //   G: t = 55*180+128 = 10028, (10028 + 39) >> 8 = 39, 180-39 = 141
        //   B: t = 195*180+128 = 35228, (35228 + 137) >> 8 = 138, 180-138 = 42
        assert_eq!(pil_cmyk_to_rgb(29, 200, 60, 180), [20, 141, 42]);
        // Raw (21, 40, 230, 96) inverts to (234, 215, 25, 159), nk = 96:
        //   R: t = 22592, (22592 + 88) >> 8 = 88, 96-88 = 8
        //   G: t = 20768, (20768 + 81) >> 8 = 81, 96-81 = 15
        //   B: t = 2528, (2528 + 9) >> 8 = 9, 96-9 = 87
        assert_eq!(pil_cmyk_to_rgb(21, 40, 230, 96), [8, 15, 87]);
        // Raw (200, 100, 50, 180): G rounds up (100*180/255 = 70.6 -> 71), where
        // a truncating product would give 70.
        assert_eq!(pil_cmyk_to_rgb(200, 100, 50, 180), [141, 71, 35]);
    }

    /// The buffer conversion is the pixel formula on every pixel, for every
    /// pixel count around and across the vector block size, tails included.
    #[test]
    fn cmyk_to_rgb_like_pil_converts_every_pixel_whatever_the_count() {
        for pixels in (0..=300).chain([511, 512, 513, 1000, 4096 + 7]) {
            let cmyk: Vec<u8> = (0..pixels * 4)
                .map(|j| ((j * 97 + 13) ^ (j >> 3)) as u8)
                .collect();
            let expected: Vec<u8> = cmyk
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| pil_cmyk_to_rgb(p[0], p[1], p[2], p[3]))
                .collect();
            assert_eq!(cmyk_to_rgb_like_pil(cmyk), expected, "{pixels} pixels");
        }
    }

    /// Every `(ink, k)` byte pair through the buffer conversion (the vector
    /// path on AArch64), against the pixel formula.
    #[test]
    fn cmyk_to_rgb_like_pil_agrees_with_the_formula_on_every_ink_and_k() {
        let cmyk: Vec<u8> = (0..=255u8)
            .flat_map(|k| {
                (0..=255u8)
                    .flat_map(move |ink| [ink, ink.wrapping_add(85), ink.wrapping_add(170), k])
            })
            .collect();
        let expected: Vec<u8> = cmyk
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| pil_cmyk_to_rgb(p[0], p[1], p[2], p[3]))
            .collect();
        assert_eq!(cmyk_to_rgb_like_pil(cmyk), expected);
    }

    /// The shortcut `MULDIV255(C, K)` is PIL's `CMYK;I` inversion followed by
    /// `cmyk2rgb`, written out literally, on every pair of bytes.
    #[test]
    fn pil_cmyk_to_rgb_equals_pils_literal_formula_on_every_byte_pair() {
        fn muldiv255(a: i32, b: i32) -> i32 {
            let t = a * b + 128;
            (t + (t >> 8)) >> 8
        }
        for k in 0..=255_u8 {
            for ink in 0..=255_u8 {
                let (ink_inverted, k_inverted) = (255 - i32::from(ink), 255 - i32::from(k));
                let nk = 255 - k_inverted;
                let literal = (nk - muldiv255(ink_inverted, nk)).clamp(0, 255);
                assert_eq!(
                    i32::from(pil_cmyk_to_rgb(ink, ink, ink, k)[0]),
                    literal,
                    "ink {ink} k {k}"
                );
            }
        }
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/images/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("fixture")
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf29ce484222325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        })
    }

    /// The three 16x16 four-component fixtures: a plain CMYK JPEG with an
    /// Adobe marker (transform 0, written by PIL), and two YCCK JPEGs
    /// (transform 2, 4:4:4 and 4:2:0, written by TurboJPEG's compressor). The
    /// expected pixels and the FNV-1a 64 hash of the RGB8 buffer are what
    /// PIL 12.3 on libjpeg-turbo 2.1.5 returns for `Image.open(f).convert("RGB")`.
    /// Needs libturbojpeg at runtime; without it the pure-Rust fallback
    /// decodes CMYK with its own DCT and upsampling, so this test skips, or
    /// fails where `SMG_REQUIRE_TURBOJPEG` says the library must be there.
    #[test]
    fn cmyk_and_ycck_jpegs_decode_to_pils_pixels() {
        if !crate::test_support::turbojpeg_or_skip("the CMYK parity test") {
            return;
        }
        type Pixel = (u32, u32, [u8; 3]);
        // (file, hash, [(x, y, rgb)])
        let cases: [(&str, u64, &[Pixel]); 3] = [
            (
                "cmyk_adobe_16x16.jpg",
                0x3aa9704dfb819025,
                &[
                    (0, 0, [220, 30, 40]),
                    (8, 0, [30, 200, 60]),
                    (0, 8, [20, 40, 230]),
                    (15, 15, [250, 240, 20]),
                ],
            ),
            (
                "ycck_adobe_16x16.jpg",
                0x0653033499c13ae5,
                &[
                    (0, 0, [220, 30, 40]),
                    (8, 0, [20, 141, 42]),
                    (0, 8, [8, 15, 87]),
                    (15, 15, [0, 0, 0]),
                ],
            ),
            (
                "ycck_adobe_16x16_420.jpg",
                0xea4a51922fe8487c,
                &[
                    (0, 0, [216, 32, 42]),
                    (7, 7, [138, 66, 67]),
                    (8, 0, [62, 119, 47]),
                    (0, 8, [23, 11, 66]),
                    (8, 7, [73, 115, 40]),
                    (15, 15, [0, 0, 0]),
                ],
            ),
        ];
        for (name, hash, pixels) in cases {
            let image = decode_jpeg_rgb(&fixture(name))
                .unwrap_or_else(|| panic!("{name}: libjpeg-turbo did not decode it"));
            let rgb = image.to_rgb8();
            assert_eq!((rgb.width(), rgb.height()), (16, 16), "{name}");
            for &(x, y, expected) in pixels {
                assert_eq!(rgb.get_pixel(x, y).0, expected, "{name} at ({x}, {y})");
            }
            assert_eq!(fnv1a(rgb.as_raw()), hash, "{name}: not PIL's pixels");
        }
    }

    /// A 16x8 baseline JPEG of one MCU with every coefficient zero, so every
    /// sample decodes to 128, its components sampled as `sampling` says (one
    /// packed byte per component, the horizontal factor in the high nibble).
    /// Built here rather than stored: a byte table would be test data.
    ///
    /// The scan holds one block per sampling unit, which is the scan's count
    /// only when the components share one interleaved MCU (two or more of
    /// them) at least 16 columns wide (a horizontal factor of 2 or more
    /// somewhere). 4:4:4 takes two MCUs, so half the blocks go unwritten:
    /// libjpeg warns of the premature end, fills them with zeros and keeps
    /// the frame, grey like the written half, so no pixel check would tell.
    /// A single component's scan is not interleaved and takes the frame's
    /// two blocks whatever its factors: one short at 1x1, exact at 2x1, two
    /// over at 2x2. The builder refuses such lists.
    fn unequal_sampling_jpeg(sampling: &[u8]) -> Vec<u8> {
        assert!(
            sampling.len() >= 2 && sampling.iter().any(|&factors| factors >> 4 >= 2),
            "one MCU of 16x8 takes two or more components, one with a horizontal \
             factor of 2 or more: not {sampling:02X?}"
        );
        let count = sampling.len();
        let mut jpeg = vec![0xFF, 0xD8];
        // DQT: one table of ones.
        jpeg.extend([0xFF, 0xDB, 0x00, 0x43, 0x00]);
        jpeg.extend([1u8; 64]);
        // SOF0: 8 bits, 8 rows, 16 columns, the components with their
        // sampling, all on quantization table 0.
        jpeg.extend([
            0xFF,
            0xC0,
            0x00,
            8 + 3 * count as u8,
            0x08,
            0x00,
            0x08,
            0x00,
            0x10,
        ]);
        jpeg.push(count as u8);
        for (id, &factors) in (1..).zip(sampling) {
            jpeg.extend([id, factors, 0]);
        }
        // DHT: a DC table and an AC table with one code each, for symbol 0 (a
        // zero difference; the end of a block).
        for class in [0x00, 0x10] {
            jpeg.extend([0xFF, 0xC4, 0x00, 0x14, class, 1]);
            jpeg.extend([0u8; 15]);
            jpeg.push(0x00);
        }
        // SOS: every component on tables 0 and 0, the whole spectrum.
        jpeg.extend([0xFF, 0xDA, 0x00, 6 + 2 * count as u8, count as u8]);
        for id in 1..=count as u8 {
            jpeg.extend([id, 0]);
        }
        jpeg.extend([0, 0x3F, 0]);
        // One block per sampling unit of each component, each a zero
        // difference and an end of block: two zero bits a block, padded
        // with ones to the byte.
        let blocks: usize = sampling
            .iter()
            .map(|&factors| usize::from(factors >> 4) * usize::from(factors & 0x0F))
            .sum();
        let bits = 2 * blocks;
        let mut scan = vec![0u8; bits.div_ceil(8)];
        let padding = scan.len() * 8 - bits;
        if let Some(last) = scan.last_mut() {
            *last = (1u8 << padding) - 1;
        }
        jpeg.extend(scan);
        jpeg.extend([0xFF, 0xD9]);
        jpeg
    }

    /// The four-component file: its first component sampled 2x2 and the other
    /// three 1x1, the shape a CMYK file written with chroma subsampling takes,
    /// which libjpeg-turbo 2.x's header call cannot name.
    fn unequal_sampling_cmyk_jpeg() -> Vec<u8> {
        unequal_sampling_jpeg(&[0x22, 0x11, 0x11, 0x11])
    }

    /// A list the builder cannot write is refused rather than written short:
    /// three components sampled 1x1 each take two MCUs across the 16 columns.
    #[test]
    #[should_panic(expected = "one MCU of 16x8 takes two or more components")]
    fn the_builder_refuses_a_list_whose_mcu_is_narrower_than_the_frame() {
        unequal_sampling_jpeg(&[0x11, 0x11, 0x11]);
    }

    /// A single component's scan is not interleaved: its block count is the
    /// component's own, not one MCU's.
    #[test]
    #[should_panic(expected = "one MCU of 16x8 takes two or more components")]
    fn the_builder_refuses_a_single_component() {
        unequal_sampling_jpeg(&[0x22]);
    }

    /// The reader finds the frame header behind the tables, in a file written
    /// by a compressor as in one built by hand, steps over fill bytes, and
    /// reads nothing from a file whose scan comes first, whose height is zero,
    /// that ends before the frame header, or that is no JPEG at all.
    #[test]
    fn the_frame_header_reads_the_dimensions_and_the_component_count() {
        let jpeg = unequal_sampling_cmyk_jpeg();
        let frame = FrameHeader {
            width: 16,
            height: 8,
            components: 4,
        };
        assert_eq!(frame_header(&jpeg), Some(frame));
        assert_eq!(
            frame_header(&fixture("ycck_adobe_16x16_420.jpg")),
            Some(FrameHeader {
                width: 16,
                height: 16,
                components: 4
            })
        );
        // SOI and the quantization table are 71 bytes; the frame header follows.
        assert_eq!(
            frame_header(&jpeg[..71]),
            None,
            "cut before the frame header"
        );
        assert_eq!(
            frame_header(&jpeg[..80]),
            None,
            "cut inside the frame header"
        );
        assert_eq!(frame_header(b"not a jpeg"), None);
        // Fill bytes before a marker are stepped over.
        let mut filled = jpeg.clone();
        filled.splice(2..2, [0xFF, 0xFF]);
        assert_eq!(
            frame_header(&filled),
            Some(frame),
            "fill bytes before a marker"
        );
        // A standalone marker (TEM) carries no length and is stepped over.
        let mut standalone = jpeg.clone();
        standalone.splice(2..2, [0xFF, 0x01]);
        assert_eq!(
            frame_header(&standalone),
            Some(frame),
            "a standalone marker before the frame header"
        );
        // A zero height (a file that defines it by a DNL segment) is no frame
        // header: the height sits five bytes into the segment at 71.
        let mut dnl = jpeg.clone();
        dnl[76..78].copy_from_slice(&[0, 0]);
        assert_eq!(frame_header(&dnl), None, "a zero height");
        // The scan before any frame header.
        assert_eq!(
            frame_header(&[0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02]),
            None,
            "a scan before the frame header"
        );
    }

    /// libjpeg-turbo 2.x's header call refuses a four-component frame whose
    /// black is sampled unlike its first component; the decoder takes it, so
    /// the frame header stands in and the image decodes through the library
    /// (the pure-Rust fallback differs from it by a few levels per pixel).
    #[test]
    fn a_four_component_jpeg_with_unequal_sampling_decodes_through_turbojpeg() {
        if !crate::test_support::turbojpeg_or_skip("the unequal-sampling test") {
            return;
        }
        let image = decode_jpeg_rgb(&unequal_sampling_cmyk_jpeg())
            .expect("decoded through libjpeg-turbo, the header call notwithstanding");
        let rgb = image.to_rgb8();
        assert_eq!((rgb.width(), rgb.height()), (16, 8));
        let expected = pil_cmyk_to_rgb(128, 128, 128, 128);
        assert!(
            rgb.pixels().all(|pixel| pixel.0 == expected),
            "every sample is 128 before the inversion: {expected:?}"
        );
    }

    /// The hand-built file with two frame headers, read two ways: a stuffed
    /// zero (`FF 00`) followed by a length, the real frame header, the head of
    /// an APP1 segment, four zero bytes and a second frame header, 17 columns
    /// wide, then the tables and the scan. libjpeg discards the stuffed zero
    /// and reads the real header, then skips the APP1 segment, which spans
    /// the second; a walk that took the stuffed zero for a segment would skip
    /// the real header by the length behind it, land in the APP1 payload and
    /// read the second.
    fn two_frame_jpeg() -> Vec<u8> {
        let plain = unequal_sampling_cmyk_jpeg();
        // SOI and the quantization table are 71 bytes; the 22-byte frame
        // header follows.
        let (head, rest) = plain.split_at(71);
        let (frame, tail) = rest.split_at(22);
        let mut wide = frame.to_vec();
        wide[8] = 0x11;
        let mut jpeg = head.to_vec();
        jpeg.extend([0xFF, 0x00, 0x00, 0x1E]);
        jpeg.extend_from_slice(frame);
        jpeg.extend([0xFF, 0xE1, 0x00, 0x1C, 0, 0, 0, 0]);
        jpeg.extend_from_slice(&wide);
        jpeg.extend_from_slice(tail);
        jpeg
    }

    /// The reader steps over a stuffed zero as libjpeg does and reads the
    /// frame header the library reads: 16 columns, not the 17 of the header
    /// behind the APP1 segment.
    #[test]
    fn the_reader_steps_over_a_stuffed_zero_as_libjpeg_does() {
        assert_eq!(
            frame_header(&two_frame_jpeg()),
            Some(FrameHeader {
                width: 16,
                height: 8,
                components: 4
            })
        );
    }

    /// The two-frame file decodes to the frame the library reads, pixel for
    /// pixel the plain file's: the library warns of the stray bytes and the
    /// decode is kept, in a buffer the size of the frame it wrote. A buffer
    /// sized by the second header (17 columns) would have its last row
    /// reached by the library's 16-column rows and its tail never written.
    #[test]
    fn a_two_frame_file_decodes_the_frame_the_library_reads() {
        if !crate::test_support::warned_decode_or_skip("the two-frame test") {
            return;
        }
        let plain = decode_jpeg_rgb(&unequal_sampling_cmyk_jpeg())
            .expect("the plain file decodes")
            .to_rgb8();
        let two = decode_jpeg_rgb(&two_frame_jpeg())
            .expect("the two-frame file decodes, as the reference decoder decodes it")
            .to_rgb8();
        assert_eq!((two.width(), two.height()), (16, 8));
        assert_eq!(two.as_raw(), plain.as_raw());
    }

    /// The fallback frame is the library's dimensions with the reader's
    /// component count, and only when the reader saw the frame the library
    /// read: a reader frame of other dimensions, of two or five components,
    /// or none at all takes no buffer.
    #[test]
    fn the_fallback_frame_needs_the_reader_to_see_the_library_frame() {
        let frame = |width, height, components| {
            Some(FrameHeader {
                width,
                height,
                components,
            })
        };
        assert_eq!(fallback_frame(16, 8, frame(16, 8, 4)), Some((16, 8, true)));
        assert_eq!(fallback_frame(16, 8, frame(16, 8, 3)), Some((16, 8, false)));
        assert_eq!(fallback_frame(16, 8, frame(17, 8, 4)), None);
        assert_eq!(fallback_frame(16, 8, frame(16, 9, 4)), None);
        assert_eq!(fallback_frame(16, 8, frame(16, 8, 2)), None);
        assert_eq!(fallback_frame(16, 8, frame(16, 8, 5)), None);
        assert_eq!(fallback_frame(16, 8, None), None);
    }

    /// The three-component half of the fallback's key: a file whose second
    /// component is sampled 2x1 and the others 1x1, a pattern the header call
    /// names no subsampling for, decodes through the library as YCbCr to grey
    /// 128 on every pixel (the decoder takes any whole-number sampling ratio).
    #[test]
    fn a_three_component_jpeg_with_unequal_sampling_decodes_through_turbojpeg() {
        if !crate::test_support::turbojpeg_or_skip("the three-component test") {
            return;
        }
        let jpeg = unequal_sampling_jpeg(&[0x11, 0x21, 0x11]);
        assert_eq!(
            frame_header(&jpeg),
            Some(FrameHeader {
                width: 16,
                height: 8,
                components: 3
            }),
            "the frame header reads the three-component frame"
        );
        let image = decode_jpeg_rgb(&jpeg)
            .expect("decoded through libjpeg-turbo, the header call notwithstanding");
        let rgb = image.to_rgb8();
        assert_eq!((rgb.width(), rgb.height()), (16, 8));
        assert!(
            rgb.pixels().all(|pixel| pixel.0 == [128, 128, 128]),
            "every sample is 128: grey"
        );
    }

    /// A header the library cannot read leaves the dimensions unset, and the
    /// decode answers `None` without taking a buffer: the hand-built file with
    /// its precision set to 9 bits (no build reads it: 2.x wants its compiled
    /// 8, 3.x takes 8, 12 and 16) and its frame grown to 1024x1024, which the
    /// frame header alone would read and serve from the 4 MiB class.
    #[test]
    fn a_header_the_library_cannot_read_takes_no_buffer() {
        use crate::buffer_pool::{self, BufferPool};
        if !crate::test_support::turbojpeg_or_skip("the unreadable-header test") {
            return;
        }
        let mut jpeg = unequal_sampling_cmyk_jpeg();
        // The frame header at 71: marker, length, precision, height, width.
        jpeg[75] = 9;
        jpeg[76..80].copy_from_slice(&[0x04, 0x00, 0x04, 0x00]);
        assert_eq!(
            frame_header(&jpeg),
            Some(FrameHeader {
                width: 1024,
                height: 1024,
                components: 4
            }),
            "the frame header alone still reads the frame"
        );
        static MINE: BufferPool = BufferPool::new();
        let decoded = buffer_pool::with_pool(&MINE, || decode_jpeg_rgb(&jpeg));
        assert!(decoded.is_none(), "the library cannot read a 9-bit header");
        assert_eq!(MINE.held(), 0, "no buffer was taken for the frame");
    }

    /// A header the library read but warned about is not the sampling
    /// refusal: the fixture with two stray bytes between its frame header and
    /// the next segment ("extraneous bytes before marker") fails the header
    /// call with its dimensions and sampling filled, and the decode, which
    /// ends in the same warning, serves the fixture's pixels (the reference
    /// decoder's, pinned by the parity test).
    #[test]
    fn a_header_with_a_warning_decodes_through_the_library() {
        if !crate::test_support::warned_decode_or_skip("the warned-header test") {
            return;
        }
        let mut jpeg = fixture("ycck_adobe_16x16_420.jpg");
        // SOI, APP14 and two quantization tables are 156 bytes; the frame
        // header follows and ends at 178, where the first Huffman table
        // begins.
        assert_eq!(
            jpeg[178..180],
            [0xFF, 0xC4],
            "the fixture's first Huffman table"
        );
        jpeg.splice(178..178, [0x00, 0x00]);
        let image = decode_jpeg_rgb(&jpeg).expect("a warned header decodes through the library");
        assert_eq!(
            fnv1a(image.to_rgb8().as_raw()),
            0xea4a51922fe8487c,
            "the fixture's pixels"
        );
    }

    /// A decode the library finished with a warning is served: `small.jpg`
    /// with two stray bytes before its scan header ("extraneous bytes before
    /// marker") decodes to the clean file's pixels, the reference decoder's,
    /// where it fell back to the pure-Rust decoder before.
    #[test]
    fn a_frame_the_library_finished_with_a_warning_is_kept() {
        if !crate::test_support::warned_decode_or_skip("the warned-decode test") {
            return;
        }
        let clean = fixture("small.jpg");
        let reference = decode_jpeg_rgb(&clean)
            .expect("the clean fixture decodes")
            .to_rgb8();
        assert_eq!(
            fnv1a(reference.as_raw()),
            0x4ee5fa805d60fea5,
            "the reference decoder's pixels for the clean file"
        );
        let mut jpeg = clean;
        // The scan header sits at 609, behind the four Huffman tables.
        assert_eq!(jpeg[609..611], [0xFF, 0xDA], "the fixture's scan header");
        jpeg.splice(609..609, [0x12, 0x34]);
        let warned = decode_jpeg_rgb(&jpeg)
            .expect("the warned decode is kept")
            .to_rgb8();
        assert!(
            warned.as_raw() == reference.as_raw(),
            "the warned decode's pixels are the clean file's"
        );
    }

    /// A corrupt scan the library decodes past (one flipped byte, which ends
    /// in "extraneous bytes before marker 0xd9") is served with the reference
    /// decoder's pixels: `small.jpg` with its byte at 703 flipped, whose
    /// decode by PIL 12.3 on libjpeg-turbo hashes as pinned.
    #[test]
    fn a_corrupt_scan_decodes_to_the_reference_decoders_pixels() {
        if !crate::test_support::warned_decode_or_skip("the corrupt-scan test") {
            return;
        }
        let mut jpeg = fixture("small.jpg");
        assert_eq!(jpeg[703], 0x80, "the byte flipped");
        jpeg[703] ^= 0x5A;
        let image = decode_jpeg_rgb(&jpeg).expect("the warned decode is kept");
        assert_eq!(
            fnv1a(image.to_rgb8().as_raw()),
            0x39c68f06e78f04df,
            "not the reference decoder's pixels"
        );
    }

    /// A file whose data ends before its end-of-image marker is not served
    /// from the frame the library filled (the reference decoder refuses the
    /// file): `large.jpg` without its last two bytes answers `None`, and the
    /// buffer taken for it goes back to its class.
    #[test]
    fn a_frame_cut_before_its_end_marker_is_not_kept() {
        use crate::buffer_pool::{self, BufferPool};
        if !crate::test_support::turbojpeg_or_skip("the cut-file test") {
            return;
        }
        let jpeg = fixture("large.jpg");
        let cut = &jpeg[..jpeg.len() - 2];
        static MINE: BufferPool = BufferPool::new();
        let decoded = buffer_pool::with_pool(&MINE, || decode_jpeg_rgb(cut));
        assert!(decoded.is_none(), "a cut file is not served");
        assert_eq!(MINE.held(), 64 << 20, "the buffer went back to its class");
    }

    /// A single-scan file whose end marker is replaced by a comment segment,
    /// its payload the marker's two bytes, is served with the clean file's
    /// pixels, as the reference decoder serves it: the comment's marker
    /// ended the entropy-coded data with every row out, and the library's
    /// warned frame is kept.
    #[test]
    fn a_single_scan_file_ending_in_a_comment_is_kept() {
        if !crate::test_support::warned_decode_or_skip("the comment-tail test") {
            return;
        }
        let clean = fixture("small.jpg");
        let reference = decode_jpeg_rgb(&clean)
            .expect("the clean fixture decodes")
            .to_rgb8();
        let mut jpeg = clean[..clean.len() - 2].to_vec();
        jpeg.extend([0xFF, 0xFE, 0x00, 0x04, 0xFF, 0xD9]);
        let served = decode_jpeg_rgb(&jpeg)
            .expect("the comment tail is served")
            .to_rgb8();
        assert!(
            served.as_raw() == reference.as_raw(),
            "the served pixels are the clean file's"
        );
    }

    /// The reading of where the data ends: at the marker as the last two
    /// bytes or behind an appendage; in the data when the last two bytes are
    /// gone, the marker is cut in two, a byte follows a cut scan, or the only
    /// marker sits inside a segment before the scan; behind a comment whose
    /// payload holds the marker's two bytes, read by its length as libjpeg
    /// reads it, with or without a byte after it, and behind a comment cut
    /// inside its length; at the marker behind a comment with another
    /// payload, and behind a comment or an application segment whose length
    /// is short of its own two bytes, stepped over as nothing; a crossed
    /// scan header and a crossed table noted; refused at a marker libjpeg
    /// refuses, at a table or a restart interval with a length libjpeg
    /// refuses, and for no JPEG at all.
    #[test]
    fn scan_end_reads_the_data_behind_the_scan() {
        let jpeg = unequal_sampling_cmyk_jpeg();
        let plain = |end: End| ScanEnd {
            end,
            scan_header: false,
            table: false,
        };
        assert_eq!(scan_end(&jpeg), plain(End::Marker));
        let mut appended = jpeg.clone();
        appended.extend([0x00, 0x11, 0x22]);
        assert_eq!(
            scan_end(&appended),
            plain(End::Marker),
            "bytes after the marker"
        );
        assert_eq!(
            scan_end(&jpeg[..jpeg.len() - 2]),
            plain(End::Data),
            "the marker gone"
        );
        assert_eq!(
            scan_end(&jpeg[..jpeg.len() - 1]),
            plain(End::Data),
            "the marker cut in two"
        );
        let mut cut = jpeg[..jpeg.len() - 2].to_vec();
        cut.push(0x00);
        assert_eq!(scan_end(&cut), plain(End::Data), "a byte after a cut scan");
        // Stray bytes before the scan header, which the decoders warn of and
        // read past, are stepped over on the way to the scan; the same file
        // cut before its marker is still cut.
        let mut stray = appended.clone();
        let scan = stray
            .windows(2)
            .position(|pair| pair == [0xFF, 0xDA])
            .expect("the scan header");
        stray.splice(scan..scan, [0x12, 0x34]);
        assert_eq!(
            scan_end(&stray),
            plain(End::Marker),
            "stray bytes before the scan header"
        );
        assert_eq!(
            scan_end(&stray[..stray.len() - 5]),
            plain(End::Data),
            "the stray file cut before its marker"
        );
        // A marker inside an application segment before the scan (an embedded
        // thumbnail's end) is not the image's.
        let mut thumbnail = jpeg[..jpeg.len() - 2].to_vec();
        thumbnail.splice(2..2, [0xFF, 0xE1, 0x00, 0x04, 0xFF, 0xD9]);
        assert_eq!(
            scan_end(&thumbnail),
            plain(End::Data),
            "a marker inside a segment before the scan"
        );
        let mut comment = jpeg[..jpeg.len() - 2].to_vec();
        comment.extend([0xFF, 0xFE, 0x00, 0x04, 0xFF, 0xD9]);
        assert_eq!(
            scan_end(&comment),
            plain(End::Segment),
            "the marker bytes inside a comment's payload"
        );
        comment.push(0x00);
        assert_eq!(
            scan_end(&comment),
            plain(End::Segment),
            "the same with a trailing byte"
        );
        let mut short = jpeg[..jpeg.len() - 2].to_vec();
        short.extend([0xFF, 0xFE, 0x00]);
        assert_eq!(
            scan_end(&short),
            plain(End::Segment),
            "a comment cut inside its length"
        );
        let mut commented = jpeg[..jpeg.len() - 2].to_vec();
        commented.extend([0xFF, 0xFE, 0x00, 0x04, 0x61, 0x62, 0xFF, 0xD9]);
        assert_eq!(
            scan_end(&commented),
            plain(End::Marker),
            "a comment before the marker"
        );
        let mut nothing = jpeg[..jpeg.len() - 2].to_vec();
        nothing.extend([0xFF, 0xFE, 0x00, 0x01, 0xFF, 0xD9]);
        assert_eq!(
            scan_end(&nothing),
            plain(End::Marker),
            "a comment short of its own two bytes skips nothing"
        );
        let mut empty = jpeg[..jpeg.len() - 2].to_vec();
        empty.extend([0xFF, 0xE1, 0x00, 0x00, 0xFF, 0xD9]);
        assert_eq!(
            scan_end(&empty),
            plain(End::Marker),
            "an application segment with a zero length skips nothing"
        );
        // The scan header and the tables are noted when crossed.
        let header = &jpeg[scan..scan + 2 + 14];
        assert_eq!(&header[..2], [0xFF, 0xDA], "the fixture's scan header");
        let mut second = jpeg[..jpeg.len() - 2].to_vec();
        second.extend_from_slice(header);
        assert_eq!(
            scan_end(&second),
            ScanEnd {
                end: End::Segment,
                scan_header: true,
                table: false
            },
            "a second scan header, nothing behind it"
        );
        second.extend([0xFF, 0xD9]);
        assert_eq!(
            scan_end(&second),
            ScanEnd {
                end: End::Marker,
                scan_header: true,
                table: false
            },
            "a second scan header before the marker"
        );
        let mut table = jpeg[..jpeg.len() - 2].to_vec();
        table.extend([0xFF, 0xDD, 0x00, 0x04, 0x00, 0x10]);
        assert_eq!(
            scan_end(&table),
            ScanEnd {
                end: End::Segment,
                scan_header: false,
                table: true
            },
            "a restart interval behind the scan"
        );
        let mut refused = jpeg[..jpeg.len() - 2].to_vec();
        refused.extend([0xFF, 0x02, 0x00, 0x04, 0x61, 0x62, 0xFF, 0xD9]);
        assert_eq!(
            scan_end(&refused),
            plain(End::Refused),
            "a marker libjpeg refuses behind the scan"
        );
        let mut bad_table = jpeg[..jpeg.len() - 2].to_vec();
        bad_table.extend([0xFF, 0xC4, 0x00, 0x01, 0xFF, 0xD9]);
        assert_eq!(
            scan_end(&bad_table).end,
            End::Refused,
            "a table length libjpeg refuses"
        );
        let mut bad_interval = jpeg[..jpeg.len() - 2].to_vec();
        bad_interval.extend([0xFF, 0xDD, 0x00, 0x03, 0x00, 0xFF, 0xD9]);
        assert_eq!(
            scan_end(&bad_interval).end,
            End::Refused,
            "a restart interval length libjpeg refuses"
        );
        assert_eq!(scan_end(b"not a jpeg"), plain(End::Refused));
    }

    /// The frame read for its scan count: a sequential frame whose first
    /// scan names every component is one scan; a progressive frame, a first
    /// scan naming fewer components than the frame, and a file with no scan
    /// header are not.
    #[test]
    fn single_scan_reads_the_frame_and_its_first_scan() {
        let jpeg = unequal_sampling_cmyk_jpeg();
        assert!(single_scan(&jpeg));
        let frame = jpeg
            .windows(2)
            .position(|pair| pair[0] == 0xFF && is_frame_header(pair[1]))
            .expect("the frame header");
        let mut progressive = jpeg.clone();
        progressive[frame + 1] = 0xC2;
        assert!(!single_scan(&progressive), "a progressive frame");
        let scan = jpeg
            .windows(2)
            .position(|pair| pair == [0xFF, 0xDA])
            .expect("the scan header");
        let mut partial = jpeg.clone();
        partial[scan + 4] -= 1;
        assert!(!single_scan(&partial), "a first scan short of a component");
        assert!(!single_scan(&jpeg[..scan]), "no scan header");
    }

    /// The reference decoder's rule over the end of a file: served at the
    /// marker; behind a comment, served when the frame is one scan and not
    /// when it is progressive; not served in the data, behind zero bytes in
    /// place of the marker, or at a refused marker. A second scan header or
    /// a table behind a single-scan frame's scan is not served, marker or no
    /// marker, where a progressive frame's is.
    #[test]
    fn end_is_served_follows_the_reference_decoders_rule() {
        let jpeg = unequal_sampling_cmyk_jpeg();
        assert!(end_is_served(&jpeg));
        let scan = jpeg
            .windows(2)
            .position(|pair| pair == [0xFF, 0xDA])
            .expect("the scan header");
        let header = jpeg[scan..scan + 2 + 14].to_vec();
        let mut second = jpeg[..jpeg.len() - 2].to_vec();
        second.extend_from_slice(&header);
        assert!(
            !end_is_served(&second),
            "a second scan header behind a single-scan frame's scan"
        );
        second.extend([0xFF, 0xD9]);
        assert!(!end_is_served(&second), "the same before the marker");
        let frame = second
            .windows(2)
            .position(|pair| pair[0] == 0xFF && is_frame_header(pair[1]))
            .expect("the frame header");
        second[frame + 1] = 0xC2;
        assert!(
            end_is_served(&second),
            "a progressive frame's second scan header before the marker"
        );
        let mut table = jpeg[..jpeg.len() - 2].to_vec();
        table.extend([0xFF, 0xDD, 0x00, 0x04, 0x00, 0x10]);
        assert!(!end_is_served(&table), "a table behind the scan, no marker");
        table.extend([0xFF, 0xD9]);
        assert!(
            !end_is_served(&table),
            "a table behind a single-scan frame's scan, before the marker"
        );
        let frame = table
            .windows(2)
            .position(|pair| pair[0] == 0xFF && is_frame_header(pair[1]))
            .expect("the frame header");
        table[frame + 1] = 0xC2;
        assert!(
            end_is_served(&table),
            "a progressive frame's table before the marker"
        );
        let mut comment = jpeg[..jpeg.len() - 2].to_vec();
        comment.extend([0xFF, 0xFE, 0x00, 0x04, 0xFF, 0xD9]);
        assert!(
            end_is_served(&comment),
            "a single-scan file ending in a comment"
        );
        let frame = comment
            .windows(2)
            .position(|pair| pair[0] == 0xFF && is_frame_header(pair[1]))
            .expect("the frame header");
        comment[frame + 1] = 0xC2;
        assert!(!end_is_served(&comment), "the same file progressive");
        assert!(!end_is_served(&jpeg[..jpeg.len() - 2]), "the marker gone");
        let mut padded = jpeg[..jpeg.len() - 2].to_vec();
        padded.extend([0; 64]);
        assert!(!end_is_served(&padded), "zero bytes in place of the marker");
        let mut refused = jpeg[..jpeg.len() - 2].to_vec();
        refused.extend([0xFF, 0x02, 0x00, 0x04, 0x61, 0x62, 0xFF, 0xD9]);
        assert!(!end_is_served(&refused), "a marker libjpeg refuses");
    }

    /// The hand-built file cut to two components and grown to 1024x1024.
    fn two_component_jpeg() -> Vec<u8> {
        let mut jpeg = unequal_sampling_cmyk_jpeg();
        // The frame header at 71: its length 20 becomes 14, the count 4
        // becomes 2, the height and width grow, and the third and fourth
        // component specs (six bytes from 87) go.
        jpeg[74] = 0x0E;
        jpeg[76..80].copy_from_slice(&[0x04, 0x00, 0x04, 0x00]);
        jpeg[80] = 2;
        jpeg.drain(87..93);
        // The scan header, now at 131: its length 14 becomes 10, the count 4
        // becomes 2, and the third and fourth component selectors (four bytes
        // from 140) go.
        assert_eq!(jpeg[131..133], [0xFF, 0xDA], "the scan header");
        jpeg[134] = 0x0A;
        jpeg[135] = 2;
        jpeg.drain(140..144);
        jpeg
    }

    /// A frame of two components is refused on its sampling like the
    /// four-component file, and the decoder has no colour conversion for it:
    /// the hand-built file cut to two components and grown to 1024x1024
    /// answers `None` with no buffer taken, where the frame header alone
    /// would serve the 4 MiB class.
    #[test]
    fn a_two_component_frame_takes_no_buffer() {
        use crate::buffer_pool::{self, BufferPool};
        if !crate::test_support::turbojpeg_or_skip("the two-component test") {
            return;
        }
        let jpeg = two_component_jpeg();
        assert_eq!(
            frame_header(&jpeg),
            Some(FrameHeader {
                width: 1024,
                height: 1024,
                components: 2
            }),
            "the frame header reads the two-component frame"
        );
        static MINE: BufferPool = BufferPool::new();
        let decoded = buffer_pool::with_pool(&MINE, || decode_jpeg_rgb(&jpeg));
        assert!(
            decoded.is_none(),
            "two components decode through no path here"
        );
        assert_eq!(MINE.held(), 0, "no buffer was taken for the frame");
    }

    /// A fatal error after a warning in the decode call is not kept, though
    /// the library's reading says warning for the whole call: `large.jpg`
    /// with two stray bytes before its scan header (the warning) and its
    /// first quantization table renumbered 3 (table 0, which every component
    /// names, is then undefined: a fatal error at the start of the decode,
    /// before any row is written) answers `None`, and the buffer goes back
    /// to its class instead of being served with another frame's bytes.
    #[test]
    fn a_fatal_error_after_a_warning_in_the_decode_is_not_kept() {
        use crate::buffer_pool::{self, BufferPool};
        if !crate::test_support::warned_decode_or_skip("the warned-then-fatal decode test") {
            return;
        }
        let mut jpeg = fixture("large.jpg");
        assert_eq!(
            jpeg[20..25],
            [0xFF, 0xDB, 0x00, 0x43, 0x00],
            "the fixture's first quantization table, number 0"
        );
        jpeg[24] = 0x03;
        assert_eq!(jpeg[609..611], [0xFF, 0xDA], "the fixture's scan header");
        jpeg.splice(609..609, [0x12, 0x34]);
        static MINE: BufferPool = BufferPool::new();
        let decoded = buffer_pool::with_pool(&MINE, || decode_jpeg_rgb(&jpeg));
        assert!(
            decoded.is_none(),
            "a frame the library never wrote is not served"
        );
        assert_eq!(MINE.held(), 64 << 20, "the buffer went back to its class");
    }

    /// The same through the header call: the two-component file with two
    /// stray bytes before its first Huffman table warns, then has its
    /// sampling refused after the dimensions are filled, so the call reads
    /// as warned with the sampling unclassified; the frame takes the
    /// component arm and no buffer, instead of a decode that fails before
    /// any row is written and a buffer served whole.
    #[test]
    fn a_warned_header_with_the_sampling_refused_takes_no_buffer() {
        use crate::buffer_pool::{self, BufferPool};
        if !crate::test_support::turbojpeg_or_skip("the warned-then-refused header test") {
            return;
        }
        let mut jpeg = two_component_jpeg();
        assert_eq!(jpeg[87..89], [0xFF, 0xC4], "the first Huffman table");
        jpeg.splice(87..87, [0x12, 0x34]);
        static MINE: BufferPool = BufferPool::new();
        let decoded = buffer_pool::with_pool(&MINE, || decode_jpeg_rgb(&jpeg));
        assert!(
            decoded.is_none(),
            "two components decode through no path here"
        );
        assert_eq!(MINE.held(), 0, "no buffer was taken for the frame");
    }

    /// A file with stray bytes before its scan header and bytes after its
    /// end-of-image marker, which the reference decoder loads, is served
    /// with the clean file's pixels: the marker walk steps over the stray
    /// bytes on its way to the scan, behind which the marker is found.
    #[test]
    fn a_warned_file_with_an_appendage_is_kept() {
        if !crate::test_support::warned_decode_or_skip("the warned-appendage test") {
            return;
        }
        let clean = fixture("small.jpg");
        let reference = decode_jpeg_rgb(&clean)
            .expect("the clean fixture decodes")
            .to_rgb8();
        let mut jpeg = clean;
        assert_eq!(jpeg[609..611], [0xFF, 0xDA], "the fixture's scan header");
        jpeg.splice(609..609, [0x12, 0x34]);
        jpeg.extend([0x00, 0x11, 0x22]);
        let warned = decode_jpeg_rgb(&jpeg)
            .expect("the warned file with an appendage is kept")
            .to_rgb8();
        assert!(
            warned.as_raw() == reference.as_raw(),
            "the warned decode's pixels are the clean file's"
        );
    }

    /// The policy reads its two values in any case and spacing, names an
    /// unknown one, and round-trips through serde as its lowercase name.
    #[test]
    fn turbojpeg_policy_parses_its_two_values_and_names_an_unknown_one() {
        assert_eq!(TurboJpegPolicy::default(), TurboJpegPolicy::Prefer);
        assert_eq!(" Require ".parse(), Ok(TurboJpegPolicy::Require));
        assert_eq!("PREFER".parse(), Ok(TurboJpegPolicy::Prefer));
        assert_eq!(TurboJpegPolicy::parse("required"), None);
        assert_eq!(
            "required".parse::<TurboJpegPolicy>().unwrap_err(),
            "invalid TurboJPEG policy 'required'; expected prefer|require"
        );
        assert_eq!(TurboJpegPolicy::Require.to_string(), "require");
        assert_eq!(
            serde_json::to_string(&TurboJpegPolicy::Require).unwrap(),
            "\"require\""
        );
        assert_eq!(
            serde_json::from_str::<TurboJpegPolicy>("\"prefer\"").unwrap(),
            TurboJpegPolicy::Prefer
        );
        assert!(serde_json::from_str::<TurboJpegPolicy>("\"Require\"").is_err());
    }

    /// `require` with a loader that found nothing is the one refusal, with
    /// the library named; `prefer` never refuses, and `require` with the
    /// library loaded passes.
    #[test]
    fn require_refuses_without_the_library_and_prefer_never_does() {
        // A loader given a name nothing resolves: the "absent" case, stated
        // rather than depending on what this host has installed.
        let absent = load_turbojpeg_from(&["libturbojpeg-absent-for-this-test.so.0"]);
        assert!(absent.is_none());
        let available = absent.is_some();
        let error = enforce(TurboJpegPolicy::Require, available).unwrap_err();
        assert_eq!(error, TurboJpegUnavailable);
        let message = error.to_string();
        assert!(message.contains("libturbojpeg.so.0"), "{message}");
        assert!(message.contains("`require`"), "{message}");
        assert_eq!(enforce(TurboJpegPolicy::Prefer, available), Ok(()));
        assert_eq!(enforce(TurboJpegPolicy::Require, true), Ok(()));
        assert_eq!(enforce(TurboJpegPolicy::Prefer, true), Ok(()));
        // The process-wide check agrees with what this process's loader found.
        assert_eq!(
            enforce_policy(TurboJpegPolicy::Require).is_ok(),
            turbojpeg_available()
        );
        assert_eq!(enforce_policy(TurboJpegPolicy::Prefer), Ok(()));
    }
}

#[cfg(test)]
mod buffer_pool_tests {
    use super::*;
    use crate::buffer_pool::{self, BufferPool};

    /// The decoded frame is a pooled buffer: the second decode of a frame
    /// takes the buffer the first one returned and writes the same pixels
    /// into it (the library writes every pixel of the frame it reports).
    #[test]
    fn a_decode_reuses_the_returned_buffer_and_writes_the_same_pixels() {
        if !crate::test_support::turbojpeg_or_skip("the pooled decode test") {
            return;
        }
        static MINE: BufferPool = BufferPool::new();
        buffer_pool::with_pool(&MINE, || {
            // 4000x3000: 36 MB of pixels, the 64 MiB class.
            let bytes = std::fs::read(format!(
                "{}/tests/fixtures/images/large.jpg",
                env!("CARGO_MANIFEST_DIR")
            ))
            .expect("fixture");
            let first = decode_jpeg_rgb(&bytes).expect("decoded");
            let address = first.as_bytes().as_ptr();
            let reference = first.as_bytes().to_vec();
            buffer_pool::recycle_image(first);
            assert_eq!(MINE.held(), 64 << 20);
            let second = decode_jpeg_rgb(&bytes).expect("decoded");
            assert_eq!(
                second.as_bytes().as_ptr(),
                address,
                "the second decode took the first's buffer"
            );
            assert!(
                second.as_bytes() == reference.as_slice(),
                "the pixels differ on the reused buffer"
            );
        });
    }

    /// A decode the library refuses (a truncated file ends with the premature
    /// end-of-file warning, which the binding reads as a failure) returns its
    /// buffer to the pool instead of freeing it, so the next decode of that
    /// class is warm.
    #[test]
    fn a_failed_decode_returns_its_buffer_to_the_pool() {
        if !crate::test_support::turbojpeg_or_skip("the failed-decode pool test") {
            return;
        }
        static MINE: BufferPool = BufferPool::new();
        buffer_pool::with_pool(&MINE, || {
            let bytes = std::fs::read(format!(
                "{}/tests/fixtures/images/large.jpg",
                env!("CARGO_MANIFEST_DIR")
            ))
            .expect("fixture");
            let truncated = &bytes[..bytes.len() / 2];
            assert_eq!(MINE.held(), 0);
            assert!(
                decode_jpeg_rgb(truncated).is_none(),
                "a truncated file decodes to nothing"
            );
            assert_eq!(
                MINE.held(),
                64 << 20,
                "the failed decode's buffer is back in its class"
            );
        });
    }
}
