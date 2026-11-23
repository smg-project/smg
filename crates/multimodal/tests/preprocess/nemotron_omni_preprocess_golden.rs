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
//! the float resize order and the pure-Rust FFT. The fixture holds pixels
//! packed (`(patches, 3 * 16 * 16)` rows); the reference-layout processor's
//! rows carry the same values in the same order.
#![allow(clippy::expect_used, clippy::panic)]
#![expect(clippy::print_stdout, reason = "integration tests: diagnostic output")]

use std::f64::consts::PI;

use llm_multimodal::{
    audio::{DecodedAudio, ParakeetAudioProcessor},
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    vision::{
        processors::{nemotron_omni::NemotronGeometry, NemotronOmniProcessor},
        transforms::rgb_bytes,
        VideoClipOptions, VisionPreProcessor,
    },
    RgbFrameRef, VideoSamplingInfo,
};
use serde::Deserialize;

use super::make_seeded_image;

/// Normalized pixels: the float resize differs from torch's only in summation order (~1e-6).
const IMAGE_TOL: f64 = 1e-5;
/// Log-mel features: pure-Rust `rustfft` against the torch STFT.
const AUDIO_TOL: f64 = 2e-3;
/// The checkpoint's patch side, in pixels.
const PATCH: usize = 16;

const GOLDEN: &str = include_str!("../fixtures/golden/nemotron_omni_preprocess.json");

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
    let reference = reference_processor();
    for case in &golden.images {
        let (name, image) = match &case.source {
            Source::Seed {
                width,
                height,
                seed,
            } => (
                format!("seed {seed} {width}x{height}"),
                make_seeded_image(*width, *height, *seed),
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
        let out = processor.preprocess(&[&image]).expect("image preprocesses");
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

        // The reference layout holds the same patches as `(patches, 3 * p * p)`.
        let out = reference.preprocess(&[&image]).expect("image preprocesses");
        assert_eq!(
            out.feature_token_counts,
            vec![case.tokens],
            "{name}: reference tokens"
        );
        let [height, width] = [case.rows, case.cols].map(|v| v as u32 * PATCH as u32);
        assert_eq!(
            out.encoder_input.shape(),
            [(case.rows * case.cols) as usize, 3 * PATCH * PATCH],
            "{name}: reference shape"
        );
        assert!(
            matches!(out.model_specific.get("imgs_sizes"), Some(ModelSpecificValue::TupleVec(sizes)) if sizes == &[(height, width)]),
            "{name}: reference size"
        );
        assert_per_item(
            &name,
            &out,
            "image_num_patches",
            (case.rows * case.cols) as usize,
        );
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
    let reference = reference_processor();
    for case in &golden.videos {
        let VideoSource {
            width,
            height,
            seeds,
        } = &case.source;
        let name = format!("video {width}x{height} x{}", seeds.len());
        let frames: Vec<_> = seeds
            .iter()
            .map(|&seed| make_seeded_image(*width, *height, seed))
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
            .preprocess_video_rgb(&refs, &VideoClipOptions::sampled_at(2.0))
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

        // The reference layout holds the same frames as `(frames, 3, H, W)`.
        let sampling = VideoSamplingInfo {
            source_fps: 2.0,
            frame_indices: (0..refs.len()).collect(),
        };
        let out = reference
            .preprocess_video_rgb_sampled(
                &refs,
                &VideoClipOptions::sampled_at(2.0),
                Some(&sampling),
            )
            .expect("video preprocesses");
        assert_eq!(
            out.feature_token_counts,
            vec![case.tokens],
            "{name}: reference tokens"
        );
        let [frames, height, width] = [0, 1, 2].map(|i| case.grid[i] as usize);
        assert_eq!(
            out.encoder_input.shape(),
            [frames, 3, height * PATCH, width * PATCH],
            "{name}: reference shape"
        );
        let values = out.encoder_input.as_f32();
        compare(
            &name,
            &packed(values.as_slice().expect("standard layout"), values.shape()),
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
        let decoded = DecodedAudio {
            samples: audio_pcm(case.seconds, case.gain),
            sample_rate: 16000,
        };
        let out = processor
            .preprocess_decoded_clips(&[&decoded])
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

/// The reference-layout processor for the checkpoint geometry the fixture was generated for.
fn reference_processor() -> NemotronOmniProcessor {
    NemotronOmniProcessor::reference(
        NemotronGeometry::from_model_config(&serde_json::json!({
            "patch_size": 16, "downsample_ratio": 0.5, "force_image_size": 512,
            "norm_mean": [0.48145466, 0.4578275, 0.40821073],
            "norm_std": [0.26862954, 0.26130258, 0.27577711],
            "vision_config": {
                "args": {"min_num_patches": 1024, "max_num_patches": 13312},
                "video_target_num_patches": 1024,
                "video_temporal_patch_size": 2,
                "video_maintain_aspect_ratio": true
            }
        }))
        .expect("checkpoint geometry"),
    )
}

/// `(n, 3, H, W)` pixels as the engine's `DynamicResolutionImageTiler.stack`
/// packs them: `c (py yy) (px xx) -> (py px) (c yy xx)` per image.
fn packed(values: &[f32], shape: &[usize]) -> Vec<f32> {
    let &[n, c, h, w] = shape else {
        panic!("NCHW pixels, got {shape:?}")
    };
    let mut out = Vec::with_capacity(values.len());
    for item in 0..n {
        for py in 0..h / PATCH {
            for px in 0..w / PATCH {
                for channel in 0..c {
                    for yy in 0..PATCH {
                        let row = ((item * c + channel) * h + py * PATCH + yy) * w + px * PATCH;
                        out.extend_from_slice(&values[row..row + PATCH]);
                    }
                }
            }
        }
    }
    out
}
