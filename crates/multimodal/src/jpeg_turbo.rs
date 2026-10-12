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
//! This module is the crate's only FFI surface, so it locally overrides the
//! workspace-wide `unsafe_code = "deny"` for the C bindings.
#![allow(unsafe_code)]

use std::{
    os::raw::{c_int, c_uchar, c_ulong, c_void},
    sync::OnceLock,
};

use image::{DynamicImage, RgbImage};
use libloading::{Library, Symbol};

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

/// Resolved TurboJPEG entry points. Holds the loaded `Library` so the function
/// pointers stay valid for the process lifetime.
struct TurboJpeg {
    _lib: Library,
    init: TjInitDecompress,
    header: TjDecompressHeader3,
    decompress: TjDecompress2,
    destroy: TjDestroy,
}

// The function pointers are plain C entry points with no shared mutable state;
// the library handle is kept alive for the process and never mutated.
unsafe impl Send for TurboJpeg {}
unsafe impl Sync for TurboJpeg {}

fn load_turbojpeg() -> Option<TurboJpeg> {
    // Try the runtime soname first (shipped by the runtime package), then the
    // dev symlink and common macOS names.
    const CANDIDATES: &[&str] = &[
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "libturbojpeg.0.dylib",
        "libturbojpeg.dylib",
    ];
    // SAFETY: loading a system shared library by name; we only resolve the four
    // documented TurboJPEG symbols below and keep the handle for their lifetime.
    let lib = CANDIDATES
        .iter()
        .find_map(|name| unsafe { Library::new(name) }.ok())?;
    // SAFETY: each symbol is resolved against the just-loaded library with the
    // signature documented by the TurboJPEG API. We copy the bare function
    // pointers out (dropping the borrowing `Symbol`s) and keep `lib` alive in
    // the returned struct, so the pointers remain valid.
    let (init, header, decompress, destroy) = unsafe {
        let init: Symbol<TjInitDecompress> = lib.get(b"tjInitDecompress\0").ok()?;
        let header: Symbol<TjDecompressHeader3> = lib.get(b"tjDecompressHeader3\0").ok()?;
        let decompress: Symbol<TjDecompress2> = lib.get(b"tjDecompress2\0").ok()?;
        let destroy: Symbol<TjDestroy> = lib.get(b"tjDestroy\0").ok()?;
        (*init, *header, *decompress, *destroy)
    };
    Some(TurboJpeg {
        _lib: lib,
        init,
        header,
        decompress,
        destroy,
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
/// way, pixel by pixel, in place: pixel `i` is read from `4i..4i+4` before
/// `3i..3i+3` is written, so no unread byte is overwritten, and the decoded
/// buffer stays the one allocation `MAX_DECODED_BYTES` bounds.
fn cmyk_to_rgb_like_pil(mut buf: Vec<u8>) -> Vec<u8> {
    let pixels = buf.len() / 4;
    for i in 0..pixels {
        let [c, m, y, k] = [buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]];
        buf[3 * i..3 * i + 3].copy_from_slice(&pil_cmyk_to_rgb(c, m, y, k));
    }
    buf.truncate(pixels * 3);
    buf
}

/// Whether libjpeg-turbo was found, so JPEGs decode to PIL's pixels.
pub fn turbojpeg_available() -> bool {
    turbojpeg().is_some()
}

/// True if `bytes` start with the JPEG SOI marker.
pub fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

/// Decode a JPEG to an RGB8 `DynamicImage` via libjpeg-turbo (PIL-compatible
/// defaults). A CMYK or YCCK JPEG is decompressed to CMYK, the only output
/// TurboJPEG offers for it, and converted with [`pil_cmyk_to_rgb`]. Returns
/// `None` on any failure — including libturbojpeg not being available at
/// runtime — so the caller can fall back to the pure-Rust decoder.
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
    // - `buf` is an owned `Vec<u8>` sized to exactly `w*h*channels`
    //   (overflow-checked) from the header dimensions, with `channels` the
    //   pixel size of the format requested (3 for `TJPF_RGB`, 4 for
    //   `TJPF_CMYK`), and is the sole alias passed to the decoder; with
    //   `pitch=0` (=> `w*channels`) the decoder writes at most `w*h*channels`
    //   bytes, so no out-of-bounds write occurs. The image is built only after
    //   `rc == 0` confirms a successful, complete write.
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
        if hdr != 0 || w <= 0 || h <= 0 {
            (tj.destroy)(handle);
            return None;
        }
        let (wu, hu) = (w as usize, h as usize);
        // A CMYK or YCCK source can only come out as CMYK; everything else
        // (YCbCr, grayscale, RGB) comes out as RGB.
        let cmyk = colorspace == TJCS_CMYK || colorspace == TJCS_YCCK;
        let (pixel_format, channels) = if cmyk { (TJPF_CMYK, 4) } else { (TJPF_RGB, 3) };
        // Guard against absurd dimensions before allocating.
        let nbytes = match decoded_buffer_len(wu, hu, channels) {
            Some(n) => n,
            None => {
                (tj.destroy)(handle);
                return None;
            }
        };
        let mut buf = vec![0_u8; nbytes];
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
        (tj.destroy)(handle);
        if rc != 0 {
            return None;
        }
        let rgb = if cmyk { cmyk_to_rgb_like_pil(buf) } else { buf };
        RgbImage::from_raw(w as u32, h as u32, rgb).map(DynamicImage::ImageRgb8)
    }
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
    /// decodes CMYK with its own DCT and upsampling, so this test skips.
    #[test]
    #[expect(clippy::print_stderr)]
    fn cmyk_and_ycck_jpegs_decode_to_pils_pixels() {
        if !turbojpeg_available() {
            eprintln!("libturbojpeg not found: skipping the CMYK parity test");
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
}
