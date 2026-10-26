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
