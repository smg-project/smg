//! Preprocessing integration tests: the image and audio processors against the reference
//! processors' goldens, the bit-identity fingerprints of the resize and preprocess kernels, and the
//! wire layout.

use image::{DynamicImage, RgbImage};

mod audio_mel_parity;
mod decoded_rgb_reference;
mod deepseek_v41_preprocess_golden;
mod gemma4_preprocess_golden;
mod inkling_preprocess_golden;
mod muse_spark_audio_golden;
mod muse_spark_preprocess_golden;
mod nemotron_omni_preprocess_golden;
mod preprocess_fingerprint;
mod qwen_pixel_norm_golden;
mod qwen_preprocess_golden;
mod qwen_smart_resize_golden;
mod resize_fingerprint;
mod vision_golden_tests;
mod wire_parity;

/// The seeded pattern the preprocessing goldens and their
/// `scripts/generate_*_preprocess_*.py` generators share: `R=(x*7+y*3)%256`,
/// `G=(x*5+y*11)%256`, `B=(x+y*2)%256`, each plus the seed with u8 wraparound.
pub(super) fn make_seeded_image(width: u32, height: u32, seed: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([
            seed.wrapping_add(((x * 7 + y * 3) % 256) as u8),
            seed.wrapping_add(((x * 5 + y * 11) % 256) as u8),
            seed.wrapping_add(((x + y * 2) % 256) as u8),
        ])
    }))
}

/// The 64-bit FNV-1a hash of `bytes` as sixteen hex digits: the fingerprint
/// the goldens record.
pub(super) fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// `hash` advanced over `bytes` by FNV-1a, for fingerprints built in pieces.
pub(super) fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The FNV-1a fingerprint of `values` as little-endian float32 bytes.
pub(super) fn fnv1a_f32(values: &[f32]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        hash = fnv1a_update(hash, &value.to_le_bytes());
    }
    format!("{hash:016x}")
}
