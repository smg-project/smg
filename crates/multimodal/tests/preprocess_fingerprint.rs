//! Bit-identity guard for the full Qwen3-VL preprocess (resize + normalize +
//! patchify). Pins the EXACT f32 encoder_input bytes. Any perf change to those
//! stages (parallelization) MUST keep these identical to preserve vLLM/PIL
//! parity (accuracy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]
use image::{DynamicImage, RgbImage};
use llm_multimodal::vision::{
    preprocessor_config::PreProcessorConfig, processors::Qwen3VLProcessor, VisionPreProcessor,
};

fn make(w: u32, h: u32) -> DynamicImage {
    let img = RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            ((x * 7 + y * 3) % 256) as u8,
            ((x * 5 + y * 11) % 256) as u8,
            ((x + y * 2) % 256) as u8,
        ])
    });
    DynamicImage::ImageRgb8(img)
}

fn config() -> PreProcessorConfig {
    PreProcessorConfig::from_json(
        r#"{"do_resize":true,"do_normalize":true,
            "image_mean":[0.48145466,0.4578275,0.40821073],
            "image_std":[0.26862954,0.26130258,0.27577711],
            "size":{"shortest_edge":3136,"longest_edge":12845056},
            "resample":3}"#,
    )
    .unwrap()
}

fn fnv1a_f32(data: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in data {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

const CASES: &[(u32, u32)] = &[(560, 420), (840, 560), (1280, 960)];

// Captured under serial normalize/patchify with the torchvision resize kernel of the
// reference processors (`tests/qwen_preprocess_golden.rs` pins that kernel against the
// reference itself); PARALLELIZATION MUST NOT CHANGE THESE.
const EXPECTED: &[u64] = &[0x42d1f84fd0c2be1d, 0x55d33c06eb2a74e5, 0x617d3e39f58f1c45];

fn fingerprint(w: u32, h: u32) -> (u64, usize) {
    let proc = Qwen3VLProcessor::new();
    let res = proc.preprocess(&[make(w, h)], &config()).unwrap();
    let flat = res.encoder_input_flat();
    (fnv1a_f32(flat.as_ref()), flat.len())
}

#[test]
#[ignore = "capture mode"]
fn capture_preprocess_fingerprints() {
    for (w, h) in CASES {
        let (fp, n) = fingerprint(*w, *h);
        println!("{w}x{h}: 0x{fp:016x}  (len {n})");
    }
}

#[test]
fn preprocess_bit_identity() {
    if EXPECTED.iter().all(|&v| v == 0) {
        eprintln!("EXPECTED not filled; run capture_preprocess_fingerprints");
        return;
    }
    for ((w, h), &exp) in CASES.iter().zip(EXPECTED) {
        let (fp, _) = fingerprint(*w, *h);
        assert_eq!(fp, exp, "preprocess fingerprint changed for {w}x{h}");
    }
}
