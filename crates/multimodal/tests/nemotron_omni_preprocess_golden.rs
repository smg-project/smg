//! Nemotron-H Omni image, video and audio preprocessing against a torch
//! re-implementation of vLLM's.
//!
//! The golden fixture is produced by
//! `crates/multimodal/scripts/generate_nemotron_omni_preprocess_golden.py`,
//! which re-implements vLLM's dynamic-resolution image tiler, its video frame
//! sizing and `ParakeetExtractor` with torch on inputs this test rebuilds:
//! seeded or fixture images and frames, and synthetic tones.
//! Grids, clip frames and token counts must match exactly; values are checked
//! at sampled positions and through their sums, within tolerances that cover
//! the float resize order and the pure-Rust FFT.
#![allow(clippy::expect_used, clippy::panic)]
#![expect(clippy::print_stdout, reason = "integration tests: diagnostic output")]

use std::f64::consts::PI;

use image::{DynamicImage, RgbImage};
use llm_multimodal::{
    audio::{DecodedAudio, ParakeetAudioProcessor},
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    vision::{
        preprocessor_config::PreProcessorConfig, processors::NemotronOmniProcessor,
        transforms::rgb_bytes, VisionPreProcessor,
    },
    RgbFrameRef,
};
use serde::Deserialize;

/// Normalized pixels: the float resize differs from torch's only in summation order (~1e-6).
const IMAGE_TOL: f64 = 1e-5;
/// Log-mel features: pure-Rust `rustfft` against the torch STFT.
const AUDIO_TOL: f64 = 2e-3;

const GOLDEN: &str = include_str!("fixtures/golden/nemotron_omni_preprocess.json");

#[derive(Deserialize)]
struct Golden {
    images: Vec<ImageCase>,
    videos: Vec<VideoCase>,
    audios: Vec<AudioCase>,
}

#[derive(Deserialize)]
struct Summary {
    numel: usize,
    sample_stride: usize,
    samples: Vec<f64>,
    sum: f64,
    sum_sq: f64,
}

#[derive(Deserialize)]
struct ImageCase {
    source: Source,
    rows: i64,
    cols: i64,
    tokens: usize,
    #[serde(flatten)]
    summary: Summary,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Source {
    Seed { width: u32, height: u32, seed: u8 },
    File { file: String },
}

#[derive(Deserialize)]
struct VideoCase {
    source: VideoSource,
    grid: Vec<i64>,
    tokens: usize,
    #[serde(flatten)]
    summary: Summary,
}

#[derive(Deserialize)]
struct VideoSource {
    width: u32,
    height: u32,
    seeds: Vec<u8>,
}

#[derive(Deserialize)]
struct AudioCase {
    seconds: f64,
    gain: f64,
    clip_frames: Vec<i64>,
    tokens: usize,
    #[serde(flatten)]
    summary: Summary,
}

/// The generator's `seeded_image`.
fn seeded_image(width: u32, height: u32, seed: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([
            seed.wrapping_add(((x * 7 + y * 3) % 256) as u8),
            seed.wrapping_add(((x * 5 + y * 11) % 256) as u8),
            seed.wrapping_add(((x + y * 2) % 256) as u8),
        ])
    }))
}

/// The generator's `audio_pcm`.
fn audio_pcm(seconds: f64, gain: f64) -> Vec<f32> {
    let n = (seconds * 16000.0).round() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / 16000.0;
            let tau = 2.0 * PI;
            (gain
                * (0.3 * (tau * 440.0 * t).sin()
                    + 0.2 * (tau * 1234.5 * t + 0.3).sin()
                    + 0.05 * (tau * (200.0 + 900.0 * t) * t).sin())) as f32
        })
        .collect()
}

fn compare(name: &str, values: &[f32], summary: &Summary, tol: f64) {
    assert_eq!(values.len(), summary.numel, "{name}: element count");
    let mut max_diff = 0.0_f64;
    for (k, &expected) in summary.samples.iter().enumerate() {
        let diff = (f64::from(values[k * summary.sample_stride]) - expected).abs();
        max_diff = max_diff.max(diff);
    }
    let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
    let sum_sq: f64 = values.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    println!("{name}: max sampled diff {max_diff:.3e}");
    assert!(max_diff <= tol, "{name}: sampled diff {max_diff} > {tol}");
    let scale = summary.numel as f64;
    assert!(
        (sum - summary.sum).abs() <= tol * scale,
        "{name}: sum {sum} vs {}",
        summary.sum
    );
    assert!(
        (sum_sq - summary.sum_sq).abs() <= tol * scale * 4.0,
        "{name}: sum of squares {sum_sq} vs {}",
        summary.sum_sq
    );
}

/// A per-item size the gateway cuts `encoder_input` by; each case is one item.
fn assert_per_item(name: &str, out: &PreprocessedEncoderInputs, key: &str, expected: usize) {
    match out.model_specific.get(key) {
        Some(ModelSpecificValue::IntTensor { data, .. }) => {
            assert_eq!(data, &vec![expected as i64], "{name}: {key}");
        }
        _ => panic!("{name}: {key} missing"),
    }
}

#[test]
fn images_match_the_reference_tiler() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("golden fixture parses");
    let processor = NemotronOmniProcessor::new();
    for case in &golden.images {
        let (name, image) = match &case.source {
            Source::Seed {
                width,
                height,
                seed,
            } => (
                format!("seed {seed} {width}x{height}"),
                seeded_image(*width, *height, *seed),
            ),
            Source::File { file } => (
                file.clone(),
                image::open(format!(
                    "{}/tests/fixtures/images/{file}",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .expect("fixture image decodes"),
            ),
        };
        let out = processor
            .preprocess(&[image], &PreProcessorConfig::default())
            .expect("image preprocesses");
        assert_eq!(
            out.feature_token_counts,
            vec![case.tokens],
            "{name}: tokens"
        );
        match out.model_specific.get("image_grid_hw") {
            Some(ModelSpecificValue::IntTensor { data, .. }) => {
                assert_eq!(data, &vec![case.rows, case.cols], "{name}: grid");
            }
            _ => panic!("{name}: image_grid_hw missing"),
        }
        let rows = out.encoder_input.shape()[0];
        assert_per_item(&name, &out, "patches_per_image", rows);
        let values = out.encoder_input.as_f32();
        compare(
            &name,
            values.as_slice().expect("standard layout"),
            &case.summary,
            IMAGE_TOL,
        );
    }
}

#[test]
fn video_frames_match_the_reference_sizing() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("golden fixture parses");
    let processor = NemotronOmniProcessor::new();
    for case in &golden.videos {
        let VideoSource {
            width,
            height,
            seeds,
        } = &case.source;
        let name = format!("video {width}x{height} x{}", seeds.len());
        let frames: Vec<_> = seeds
            .iter()
            .map(|&seed| seeded_image(*width, *height, seed))
            .collect();
        let rgb: Vec<_> = frames.iter().map(rgb_bytes).collect();
        let refs: Vec<_> = rgb
            .iter()
            .map(|(_, _, data)| RgbFrameRef {
                width: *width,
                height: *height,
                data,
            })
            .collect();
        let out = processor
            .preprocess_video_rgb(&refs, &PreProcessorConfig::default())
            .expect("video preprocesses");
        assert_eq!(
            out.feature_token_counts,
            vec![case.tokens],
            "{name}: tokens"
        );
        match out.model_specific.get("video_grid_thw") {
            Some(ModelSpecificValue::IntTensor { data, .. }) => {
                assert_eq!(data, &case.grid, "{name}: grid");
            }
            _ => panic!("{name}: video_grid_thw missing"),
        }
        let rows = out.encoder_input.shape()[0];
        assert_per_item(&name, &out, "patches_per_video", rows);
        let values = out.encoder_input.as_f32();
        compare(
            &name,
            values.as_slice().expect("standard layout"),
            &case.summary,
            IMAGE_TOL,
        );
    }
}

#[test]
fn audio_matches_the_reference_extractor() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("golden fixture parses");
    let processor = ParakeetAudioProcessor::default();
    for case in &golden.audios {
        let name = format!("{}s x{}", case.seconds, case.gain);
        let out = processor
            .preprocess_decoded_clips(vec![DecodedAudio {
                samples: audio_pcm(case.seconds, case.gain),
                sample_rate: 16000,
            }])
            .expect("audio preprocesses");
        assert_eq!(
            out.feature_token_counts,
            vec![case.tokens],
            "{name}: tokens"
        );
        match out.model_specific.get("audio_clip_frames") {
            Some(ModelSpecificValue::IntTensor { data, .. }) => {
                assert_eq!(data, &case.clip_frames, "{name}: clip frames");
            }
            _ => panic!("{name}: audio_clip_frames missing"),
        }
        let rows = out.encoder_input.shape()[0];
        assert_per_item(&name, &out, "frames_per_item", rows);
        assert_per_item(&name, &out, "clips_per_item", case.clip_frames.len());
        let values = out.encoder_input.as_f32();
        compare(
            &name,
            values.as_slice().expect("standard layout"),
            &case.summary,
            AUDIO_TOL,
        );
    }
}
