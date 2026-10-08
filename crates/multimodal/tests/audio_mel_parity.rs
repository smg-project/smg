//! Full-tensor parity check for Qwen3 audio log-mel preprocessing.
//!
//! The golden fixture is produced by `scripts/generate_audio_mel_golden.py`
//! from HuggingFace `transformers.WhisperFeatureExtractor` -- the feature
//! extractor Qwen3-ASR / Qwen3-Omni use -- and `torchaudio.functional.resample`
//! for the non-16 kHz cases. It captures each case's input PCM and the reference
//! outputs so the pure-Rust frontend can be diffed against it here.
//!
//! Cases cover the mel path in isolation (16 kHz), the resample path
//! (44.1 kHz / 48 kHz -> 16 kHz), a sub-`n_fft` clip (reflect-pad / frame-count
//! edge), silence (the Whisper log-norm floor), and the batched path (variable
//! lengths, with `feature_attention_mask` and `audio_feature_lengths`).
//!
//! The bar is a tolerance on the full tensor, not bitwise equality: SMG uses
//! pure-Rust `rustfft` and a pure-Rust resampler, which cannot be bit-identical
//! to torch / torchaudio. Mel-only cases land ~1e-4; the band-limited resample
//! approximation is looser, so those cases carry a wider (but still real)
//! tolerance. Each case prints its observed max-abs-diff so real headroom is
//! visible in CI logs.
#![allow(clippy::expect_used, clippy::panic)]
#![expect(clippy::print_stdout, reason = "integration tests: diagnostic output")]

use std::sync::Arc;

use llm_multimodal::{
    audio::{AudioPreProcessor, DecodedAudio, Qwen3AudioProcessor},
    encoder_inputs::ModelSpecificValue,
    types::{AudioClip, AudioSource},
};
use serde::Deserialize;

/// Tolerance for the mel-only cases (no resample): pure-Rust `rustfft` vs. the
/// torch STFT. Observed max-abs-diff is ~1e-4; this leaves headroom while still
/// catching any real algorithmic regression.
const MEL_TOL: f32 = 1e-3;

/// Tolerance for the resample cases: SMG's band-limited resampler is a finite
/// approximation of `torchaudio.functional.resample`, so the resample + mel path
/// compounds two sources of error. In practice it still lands ~3e-5 (the kernel
/// matches torchaudio closely); 1e-3 keeps ample headroom for platform FFT
/// variance while remaining a real gate on resampler drift.
const RESAMPLE_TOL: f32 = 1e-3;

#[derive(Deserialize)]
struct AudioMelGolden {
    generator: String,
    reference: String,
    resampler: String,
    transformers: String,
    torch: String,
    sample_rate: usize,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    kind: String,
    sample_rate: usize,
    // Single-clip cases.
    #[serde(default)]
    pcm: Vec<f32>,
    #[serde(default)]
    mel_shape: Vec<usize>,
    mel: Vec<f32>,
    mel_sum: f64,
    // Batch case.
    #[serde(default)]
    pcm_clips: Vec<Vec<f32>>,
    #[serde(default)]
    encoder_shape: Vec<usize>,
    #[serde(default)]
    feature_attention_mask: Vec<Vec<i64>>,
    #[serde(default)]
    audio_feature_lengths: Vec<i64>,
}

/// Max-abs-diff between the Rust output and the golden, plus its flat index.
fn max_abs_diff(actual: &[f32], expected: &[f32]) -> (f32, usize) {
    assert_eq!(actual.len(), expected.len(), "value count mismatch");
    let mut max_diff = 0.0_f32;
    let mut max_at = 0usize;
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
            max_at = index;
        }
    }
    (max_diff, max_at)
}

fn tolerance_for(case: &Case) -> f32 {
    if case.sample_rate == 16_000 {
        MEL_TOL
    } else {
        RESAMPLE_TOL
    }
}

fn check_single(processor: &Qwen3AudioProcessor, case: &Case) {
    assert_eq!(
        case.mel_shape.len(),
        2,
        "{}: mel golden must be 2-D",
        case.name
    );
    assert_eq!(
        case.mel.len(),
        case.mel_shape[0] * case.mel_shape[1],
        "{}: mel value count must match declared shape",
        case.name
    );
    assert!(
        !case.pcm.is_empty(),
        "{}: golden PCM must not be empty",
        case.name
    );

    let decoded = DecodedAudio {
        samples: case.pcm.clone(),
        sample_rate: case.sample_rate,
    };
    let features = processor
        .preprocess_decoded(decoded)
        .unwrap_or_else(|error| panic!("{}: log-mel preprocessing failed: {error}", case.name));

    assert_eq!(
        features.shape(),
        case.mel_shape.as_slice(),
        "{}: log-mel shape differs from golden",
        case.name
    );

    let values = features
        .as_slice_memory_order()
        .unwrap_or_else(|| panic!("{}: log-mel output must be contiguous", case.name));
    let frames = case.mel_shape[1];
    let (diff, at) = max_abs_diff(values, &case.mel);
    let tol = tolerance_for(case);
    let actual_sum: f64 = values.iter().map(|&v| f64::from(v)).sum();

    println!(
        "audio mel parity [{:<15}]: shape={:?} max_abs_diff={diff:.3e} at (mel={}, frame={}) \
         tol={tol:.1e} rust_sum={actual_sum:.4} golden_sum={:.4}",
        case.name,
        case.mel_shape,
        at / frames,
        at % frames,
        case.mel_sum,
    );

    assert!(
        diff < tol,
        "{}: log-mel max-abs-diff {diff:.3e} exceeds tolerance {tol:.1e} at (mel={}, frame={})",
        case.name,
        at / frames,
        at % frames,
    );
    assert!(
        (actual_sum - case.mel_sum).abs() < 0.05_f64.max(case.mel.len() as f64 * 1e-4),
        "{}: log-mel sum {actual_sum:.4} differs from golden {:.4}",
        case.name,
        case.mel_sum,
    );
}

fn check_batch(processor: &Qwen3AudioProcessor, case: &Case) {
    assert_eq!(
        case.encoder_shape.len(),
        3,
        "{}: encoder shape must be 3-D",
        case.name
    );
    let batch = case.encoder_shape[0];
    let n_mels = case.encoder_shape[1];
    let max_frames = case.encoder_shape[2];
    assert_eq!(
        case.pcm_clips.len(),
        batch,
        "{}: clip count mismatch",
        case.name
    );

    // Route through the public `preprocess` entry point (Arc<AudioClip>), the
    // batched path production uses.
    let clips: Vec<Arc<AudioClip>> = case
        .pcm_clips
        .iter()
        .map(|pcm| {
            Arc::new(AudioClip::new(
                bytes::Bytes::new(),
                DecodedAudio {
                    samples: pcm.clone(),
                    sample_rate: case.sample_rate,
                },
                AudioSource::InlineBytes,
                String::new(),
            ))
        })
        .collect();

    let output = processor
        .preprocess(&clips)
        .unwrap_or_else(|error| panic!("{}: batch preprocessing failed: {error}", case.name));

    assert_eq!(
        output.encoder_input.shape(),
        case.encoder_shape.as_slice(),
        "{}: encoder_input shape differs from golden",
        case.name
    );

    let encoder_f32 = output.encoder_input.as_f32();
    let values = encoder_f32
        .as_slice_memory_order()
        .unwrap_or_else(|| panic!("{}: encoder_input must be contiguous", case.name));
    let (diff, at) = max_abs_diff(values, &case.mel);
    let tol = tolerance_for(case);
    let actual_sum: f64 = values.iter().map(|&v| f64::from(v)).sum();

    // Flat index -> (clip, mel, frame) for a [B, n_mels, max_frames] tensor.
    let clip = at / (n_mels * max_frames);
    let within = at % (n_mels * max_frames);
    println!(
        "audio mel parity [{:<15}]: shape={:?} max_abs_diff={diff:.3e} at (clip={clip}, mel={}, \
         frame={}) tol={tol:.1e} rust_sum={actual_sum:.4} golden_sum={:.4}",
        case.name,
        case.encoder_shape,
        within / max_frames,
        within % max_frames,
        case.mel_sum,
    );

    assert!(
        diff < tol,
        "{}: batch log-mel max-abs-diff {diff:.3e} exceeds tolerance {tol:.1e}",
        case.name,
    );
    assert!(
        (actual_sum - case.mel_sum).abs() < case.mel.len() as f64 * 1e-4,
        "{}: batch log-mel sum {actual_sum:.4} differs from golden {:.4}",
        case.name,
        case.mel_sum,
    );

    // Batched metadata must match Rust's per-clip semantics exactly.
    let expected_mask: Vec<i64> = case
        .feature_attention_mask
        .iter()
        .flatten()
        .copied()
        .collect();
    match output.model_specific.get("feature_attention_mask") {
        Some(ModelSpecificValue::IntTensor { data, shape }) => {
            assert_eq!(
                shape,
                &vec![batch, max_frames],
                "{}: feature_attention_mask shape",
                case.name
            );
            assert_eq!(
                data, &expected_mask,
                "{}: feature_attention_mask",
                case.name
            );
        }
        other => panic!(
            "{}: feature_attention_mask missing/typed wrong: {other:?}",
            case.name
        ),
    }

    match output.model_specific.get("audio_feature_lengths") {
        Some(ModelSpecificValue::IntTensor { data, shape }) => {
            assert_eq!(
                shape,
                &vec![batch],
                "{}: audio_feature_lengths shape",
                case.name
            );
            assert_eq!(
                data, &case.audio_feature_lengths,
                "{}: audio_feature_lengths",
                case.name
            );
        }
        other => panic!(
            "{}: audio_feature_lengths missing/typed wrong: {other:?}",
            case.name
        ),
    }

    println!(
        "audio mel parity [{:<15}]: feature_attention_mask + audio_feature_lengths match \
         (lengths={:?})",
        case.name, case.audio_feature_lengths,
    );
}

#[test]
fn qwen3_audio_log_mel_matches_transformers_golden() {
    let golden: AudioMelGolden =
        serde_json::from_str(include_str!("fixtures/golden/audio_mel_reference.json"))
            .expect("invalid checked-in audio mel golden fixture");

    assert_eq!(golden.generator, "generate_audio_mel_golden.py");
    assert_eq!(golden.reference, "transformers.WhisperFeatureExtractor");
    assert_eq!(golden.resampler, "torchaudio.functional.resample");
    assert!(
        !golden.transformers.is_empty(),
        "transformers version missing"
    );
    assert!(!golden.torch.is_empty(), "torch version missing");

    // Golden fixture is generated with Qwen3AudioParams::default() parameters.
    let processor = Qwen3AudioProcessor::new();
    let params = processor.params();
    assert_eq!(
        params.sample_rate, golden.sample_rate,
        "sample_rate mismatch"
    );
    assert_eq!(params.n_mels, golden.n_mels, "n_mels mismatch");
    assert_eq!(params.n_fft, golden.n_fft, "n_fft mismatch");
    assert_eq!(params.hop_length, golden.hop_length, "hop_length mismatch");

    assert!(!golden.cases.is_empty(), "golden must contain cases");
    let mut saw_single = false;
    let mut saw_batch = false;
    for case in &golden.cases {
        match case.kind.as_str() {
            "single" => {
                saw_single = true;
                check_single(&processor, case);
            }
            "batch" => {
                saw_batch = true;
                check_batch(&processor, case);
            }
            other => panic!("{}: unknown case kind {other:?}", case.name),
        }
    }
    assert!(
        saw_single,
        "golden must contain at least one single-clip case"
    );
    assert!(saw_batch, "golden must contain the batched case");
}
