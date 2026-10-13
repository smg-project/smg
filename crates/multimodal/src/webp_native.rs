//! Runtime (dlopen) binding to libwebp's simple decoding API for WebP decode.
//!
//! The reference image loaders decode WebP with libwebp. The pure-Rust decoder
//! the `image` crate brings produces the same pixels on the files checked, at
//! five to ten times the cost of a JPEG of the same size; libwebp decodes them
//! in the JPEG class. As with libjpeg-turbo, the library is loaded at RUNTIME
//! by name, so the crate compiles everywhere with no build-time or link-time
//! dependency: where the shared library is present the decode goes through
//! it, where it is absent `decode_webp` returns `None` and the caller falls
//! back to the pure-Rust decoder.
//!
//! Only still images take this path. An animated file (the extended header's
//! animation flag) is left to the fallback, which takes its first frame; a
//! file whose header says it carries alpha is decoded as RGBA, the rest as
//! RGB, the same image types the fallback produces for them.
//!
//! This module is one of the crate's FFI surfaces (with `jpeg_turbo` and the
//! optional `opencv_buffer`), so it locally overrides the workspace-wide
//! `unsafe_code = "deny"` for the C bindings.
#![allow(unsafe_code)]

use std::{
    os::raw::{c_int, c_uchar},
    sync::OnceLock,
};

use image::{DynamicImage, RgbImage, RgbaImage};
use libloading::{Library, Symbol};

/// Cap on the decoded buffer, the `image` crate's default limit, which the
/// pure-Rust fallback decoder already enforces.
const MAX_DECODED_BYTES: usize = 512 * 1024 * 1024;

type WebPGetInfo = unsafe extern "C" fn(*const c_uchar, usize, *mut c_int, *mut c_int) -> c_int;
type WebPDecodeInto =
    unsafe extern "C" fn(*const c_uchar, usize, *mut c_uchar, usize, c_int) -> *mut c_uchar;
type WebPGetDecoderVersion = unsafe extern "C" fn() -> c_int;

/// Resolved libwebp entry points; holds the loaded `Library` so the function
/// pointers stay valid for the process lifetime.
struct WebP {
    _lib: Library,
    info: WebPGetInfo,
    decode_rgb: WebPDecodeInto,
    decode_rgba: WebPDecodeInto,
    version: c_int,
}

// The function pointers are plain C entry points with no shared mutable state;
// the library handle is kept alive for the process and never mutated.
unsafe impl Send for WebP {}
unsafe impl Sync for WebP {}

fn load_webp() -> Option<WebP> {
    // The decoder-only library first (smaller, shipped by the runtime
    // packages), then the full library, then the macOS names.
    const CANDIDATES: &[&str] = &[
        "libwebpdecoder.so.3",
        "libwebp.so.7",
        // libwebp 0.6 (older Debian and Ubuntu bases): the same four entry points.
        "libwebp.so.6",
        "libwebpdecoder.so",
        "libwebp.so",
        "libwebpdecoder.3.dylib",
        "libwebp.7.dylib",
        "libwebp.dylib",
    ];
    // SAFETY: loading a system shared library by name; only the documented
    // libwebp symbols below are resolved and the handle is kept for their
    // lifetime.
    let lib = CANDIDATES
        .iter()
        .find_map(|name| unsafe { Library::new(name) }.ok())?;
    // SAFETY: each symbol is resolved against the just-loaded library with the
    // signature documented by the libwebp decoding API; the bare function
    // pointers are copied out and `lib` stays alive in the returned struct.
    let (info, decode_rgb, decode_rgba, version) = unsafe {
        let info: Symbol<WebPGetInfo> = lib.get(b"WebPGetInfo\0").ok()?;
        let decode_rgb: Symbol<WebPDecodeInto> = lib.get(b"WebPDecodeRGBInto\0").ok()?;
        let decode_rgba: Symbol<WebPDecodeInto> = lib.get(b"WebPDecodeRGBAInto\0").ok()?;
        let version: Symbol<WebPGetDecoderVersion> = lib.get(b"WebPGetDecoderVersion\0").ok()?;
        let version = version();
        (*info, *decode_rgb, *decode_rgba, version)
    };
    Some(WebP {
        _lib: lib,
        info,
        decode_rgb,
        decode_rgba,
        version,
    })
}

/// Process-wide cached binding, or `None` if the library is absent.
fn webp() -> Option<&'static WebP> {
    static LIB: OnceLock<Option<WebP>> = OnceLock::new();
    LIB.get_or_init(load_webp).as_ref()
}

/// Whether libwebp was found, so WebP files decode through it.
pub fn webp_available() -> bool {
    webp().is_some()
}

/// The library's version as `major.minor.patch`, when it was found.
pub fn webp_version() -> Option<String> {
    webp().map(|lib| {
        let v = lib.version;
        format!("{}.{}.{}", (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff)
    })
}

/// True if `bytes` carry the WebP container signature.
pub fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
}

/// What the container's first chunk says about the still image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// A single VP8 (lossy) or VP8L (lossless) frame; `alpha` is the lossless
    /// header's flag, or the extended header's.
    Still { alpha: bool },
    /// An animation: not this path's.
    Animated,
}

/// Reads the first chunk after the RIFF header: `VP8 ` (lossy, no alpha),
/// `VP8L` (lossless; the header's alpha bit) or `VP8X` (extended; the flags'
/// alpha and animation bits). Anything else is not a layout this path takes.
fn layout(bytes: &[u8]) -> Option<Layout> {
    if !is_webp(bytes) || bytes.len() < 20 {
        return None;
    }
    let fourcc = &bytes[12..16];
    let payload = &bytes[20..];
    match fourcc {
        b"VP8 " => Some(Layout::Still { alpha: false }),
        b"VP8L" => {
            // Signature byte 0x2f, then a little-endian 32-bit header whose
            // bit 28 is `alpha_is_used`.
            let header = u32::from_le_bytes([
                *payload.get(1)?,
                *payload.get(2)?,
                *payload.get(3)?,
                *payload.get(4)?,
            ]);
            Some(Layout::Still {
                alpha: header & (1 << 28) != 0,
            })
        }
        b"VP8X" => {
            let flags = *payload.first()?;
            if flags & 0x02 != 0 {
                Some(Layout::Animated)
            } else {
                Some(Layout::Still {
                    alpha: flags & 0x10 != 0,
                })
            }
        }
        _ => None,
    }
}

/// Byte length of a `w` x `h` buffer of `channels` bytes per pixel, or `None`
/// if it overflows or exceeds `MAX_DECODED_BYTES`.
fn decoded_buffer_len(w: usize, h: usize, channels: usize) -> Option<usize> {
    w.checked_mul(h)?
        .checked_mul(channels)
        .filter(|&n| n <= MAX_DECODED_BYTES)
}

/// Decode a still WebP to an RGB8 (or, with alpha, RGBA8) `DynamicImage` via
/// libwebp. Returns `None` on any failure, including the library not being
/// available at runtime or the file being an animation, so the caller can
/// fall back to the pure-Rust decoder.
pub fn decode_webp(bytes: &[u8]) -> Option<DynamicImage> {
    let Layout::Still { alpha } = layout(bytes)? else {
        return None;
    };
    let lib = webp()?;
    let (mut w, mut h): (c_int, c_int) = (0, 0);
    // SAFETY: `bytes` is a live slice; its pointer and length describe a valid
    // immutable region for the call, which only reads it; `w` and `h` are
    // live locals the call writes.
    let ok = unsafe { (lib.info)(bytes.as_ptr(), bytes.len(), &mut w, &mut h) };
    if ok == 0 || w <= 0 || h <= 0 {
        return None;
    }
    let (wu, hu) = (w as usize, h as usize);
    let channels = if alpha { 4 } else { 3 };
    let nbytes = decoded_buffer_len(wu, hu, channels)?;
    let mut buf = vec![0_u8; nbytes];
    let stride = c_int::try_from(wu.checked_mul(channels)?).ok()?;
    let decode = if alpha {
        lib.decode_rgba
    } else {
        lib.decode_rgb
    };
    // SAFETY: `buf` is an owned buffer of exactly `w * h * channels` bytes
    // (overflow-checked from the dimensions the library reported), passed with
    // its length and a stride of `w * channels`, so the decoder writes at most
    // `nbytes` bytes; `bytes` is only read. The image is built only after the
    // call returned the buffer's own pointer, its success signal.
    let written = unsafe {
        decode(
            bytes.as_ptr(),
            bytes.len(),
            buf.as_mut_ptr(),
            nbytes,
            stride,
        )
    };
    if written.is_null() {
        return None;
    }
    if alpha {
        RgbaImage::from_raw(w as u32, h as u32, buf).map(DynamicImage::ImageRgba8)
    } else {
        RgbImage::from_raw(w as u32, h as u32, buf).map(DynamicImage::ImageRgb8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn container(fourcc: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&((4 + 8 + payload.len()) as u32).to_le_bytes());
        out.extend_from_slice(b"WEBP");
        out.extend_from_slice(&fourcc);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn the_container_signature_and_first_chunk_are_read() {
        assert!(!is_webp(b"RIFF\0\0\0\0WAVE"));
        assert_eq!(
            layout(&container(*b"VP8 ", &[0; 16])),
            Some(Layout::Still { alpha: false })
        );
        let mut lossless = vec![0x2f_u8];
        lossless.extend_from_slice(&(1_u32 << 28).to_le_bytes());
        lossless.extend_from_slice(&[0; 8]);
        assert_eq!(
            layout(&container(*b"VP8L", &lossless)),
            Some(Layout::Still { alpha: true })
        );
        let mut opaque = vec![0x2f_u8];
        opaque.extend_from_slice(&0_u32.to_le_bytes());
        opaque.extend_from_slice(&[0; 8]);
        assert_eq!(
            layout(&container(*b"VP8L", &opaque)),
            Some(Layout::Still { alpha: false })
        );
        assert_eq!(
            layout(&container(*b"VP8X", &[0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
            Some(Layout::Still { alpha: true })
        );
        assert_eq!(
            layout(&container(*b"VP8X", &[0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
            Some(Layout::Animated)
        );
        assert_eq!(layout(&container(*b"ICCP", &[0; 16])), None);
    }

    #[test]
    fn decoded_buffer_len_caps_decoded_size() {
        assert_eq!(decoded_buffer_len(1024, 768, 3), Some(1024 * 768 * 3));
        assert_eq!(decoded_buffer_len(16383, 16383, 4), None);
    }

    /// Where the library is present, a lossless file written by the `image`
    /// crate's encoder decodes to the pixels it was given (lossless), and an
    /// alpha file to RGBA; where it is absent both return `None`.
    #[test]
    fn a_lossless_file_round_trips_through_the_library_when_present() {
        use image::{codecs::webp::WebPEncoder, ImageEncoder, Rgb, Rgba};

        let mut rgb = RgbImage::new(9, 7);
        for (x, y, pixel) in rgb.enumerate_pixels_mut() {
            *pixel = Rgb([(x * 29) as u8, (y * 37) as u8, (x * y) as u8]);
        }
        let mut encoded = Vec::new();
        WebPEncoder::new_lossless(&mut encoded)
            .write_image(rgb.as_raw(), 9, 7, image::ExtendedColorType::Rgb8)
            .expect("encode");
        assert!(is_webp(&encoded));
        match decode_webp(&encoded) {
            Some(DynamicImage::ImageRgb8(decoded)) => assert_eq!(decoded.as_raw(), rgb.as_raw()),
            Some(other) => panic!("unexpected image type {other:?}"),
            None => assert!(
                !webp_available(),
                "the library is present but declined a lossless file"
            ),
        }

        let mut rgba = RgbaImage::new(5, 4);
        for (x, y, pixel) in rgba.enumerate_pixels_mut() {
            *pixel = Rgba([(x * 50) as u8, (y * 60) as u8, 7, (x + y) as u8 * 30]);
        }
        let mut encoded = Vec::new();
        WebPEncoder::new_lossless(&mut encoded)
            .write_image(rgba.as_raw(), 5, 4, image::ExtendedColorType::Rgba8)
            .expect("encode");
        match decode_webp(&encoded) {
            Some(DynamicImage::ImageRgba8(decoded)) => assert_eq!(decoded.as_raw(), rgba.as_raw()),
            Some(other) => panic!("unexpected image type {other:?}"),
            None => assert!(!webp_available()),
        }
    }
}
