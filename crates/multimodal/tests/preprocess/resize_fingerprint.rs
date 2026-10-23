//! Bit-identity guard for resize_bicubic_pil. The fingerprints below pin the
//! EXACT byte output of the Pillow-exact BICUBIC resize. Any change to the
//! resize (e.g. parallelization for speed) MUST keep these identical — the
//! resize feeds vision-encoder input, so its output must stay bit-for-bit
//! stable to preserve vLLM/PIL parity (accuracy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]
use image::{DynamicImage, RgbImage};
use llm_multimodal::vision::transforms::resize_bicubic_pil;

fn make(w: u32, h: u32) -> DynamicImage {
    // deterministic, non-trivial structure across all 3 channels
    let img = RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            ((x * 7 + y * 3) % 256) as u8,
            ((x * 5 + y * 11) % 256) as u8,
            ((x + y * 2) % 256) as u8,
        ])
    });
    DynamicImage::ImageRgb8(img)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

const CASES: &[(u32, u32, u32, u32)] = &[
    (800, 600, 200, 150),  // downscale
    (640, 480, 336, 336),  // downscale to square
    (1280, 960, 512, 384), // downscale large
    (259, 194, 280, 196),  // slight upscale (real MMBench-ish)
    (200, 200, 700, 700),  // upscale
];

// Captured under the serial implementation; PARALLELIZATION MUST NOT CHANGE THESE.
const EXPECTED: &[u64] = &[
    0xac15afa8701536c4,
    0x9b76033374b3e1a2,
    0xf21c4a6ac5c20c83,
    0xb170a19da1087feb,
    0xf09a480918a7e2ad,
];

#[test]
#[ignore = "capture mode: prints fingerprints"]
fn capture_resize_fingerprints() {
    for (iw, ih, ow, oh) in CASES {
        let out = resize_bicubic_pil(&make(*iw, *ih), *ow, *oh);
        let h = fnv1a(out.to_rgb8().as_raw());
        println!("{iw}x{ih}->{ow}x{oh}: 0x{h:016x}");
    }
}

#[test]
fn resize_bicubic_pil_bit_identity() {
    if EXPECTED.iter().all(|&v| v == 0) {
        eprintln!("EXPECTED not yet filled; run capture_resize_fingerprints");
        return;
    }
    for ((iw, ih, ow, oh), &exp) in CASES.iter().zip(EXPECTED) {
        let out = resize_bicubic_pil(&make(*iw, *ih), *ow, *oh);
        let got = fnv1a(out.to_rgb8().as_raw());
        assert_eq!(
            got, exp,
            "resize fingerprint changed for {iw}x{ih}->{ow}x{oh}"
        );
    }
}
