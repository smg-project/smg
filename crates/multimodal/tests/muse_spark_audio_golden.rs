//! Muse Spark audio preprocessing against the reference feature extractor.
//!
//! The fixture is produced by
//! `crates/multimodal/scripts/generate_muse_spark_audio_fingerprints.py`, which
//! runs `MuseSparkFeatureExtractor` on deterministic integer-formula
//! waveforms (resampled with `torchaudio.functional.resample` when not at
//! 24 kHz) and fingerprints its `input_values` rows as float32 and as the
//! bfloat16 the gateway writes on the wire. The processor must reproduce the
//! token counts, the padded lengths and both fingerprints bit for bit.

#![allow(clippy::expect_used, clippy::panic)]

use llm_multimodal::{
    audio::{DecodedAudio, MuseSparkAudioProcessor, MUSE_SPARK_SAMPLES_PER_AUDIO_KEY},
    f32_to_bf16_bits, ModelSpecificValue, PreprocessedEncoderInputs,
};
use serde::Deserialize;

const GOLDEN: &str = include_str!("fixtures/golden/muse_spark_audio_fingerprints.json");

#[derive(Deserialize)]
struct Golden {
    sample_rate: usize,
    samples_per_token: usize,
    clips: Vec<ClipCase>,
    batches: Vec<BatchCase>,
}

#[derive(Deserialize)]
struct ClipCase {
    name: String,
    sample_rate: usize,
    samples: usize,
    seed: u32,
    amplitude: f32,
    resampled_samples: usize,
    tokens: usize,
    padded_samples: usize,
    fnv1a_f32: String,
    fnv1a_bf16: String,
}

#[derive(Deserialize)]
struct BatchCase {
    name: String,
    clips: Vec<String>,
    samples_per_audio: Vec<i64>,
    fnv1a_f32: String,
    fnv1a_bf16: String,
}

/// The generator's integer formula: exact in float32 on both sides.
fn make_waveform(count: usize, seed: u32, amplitude: f32) -> Vec<f32> {
    if seed == 0 {
        return vec![0.0; count];
    }
    (0..count)
        .map(|i| {
            let raw = ((i as u64 * 73 + seed as u64 * 977) % 65_536) as i64 - 32_768;
            raw as f32 / 32_768.0 * amplitude
        })
        .collect()
}

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn fnv1a_f32(values: &[f32]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        hash = fnv1a_update(hash, &value.to_le_bytes());
    }
    format!("{hash:016x}")
}

fn fnv1a_bf16(values: &[f32]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        hash = fnv1a_update(hash, &f32_to_bf16_bits(*value).to_le_bytes());
    }
    format!("{hash:016x}")
}

fn golden() -> Golden {
    serde_json::from_str(GOLDEN).expect("invalid checked-in Muse Spark audio fixture")
}

fn decoded(case: &ClipCase) -> DecodedAudio {
    DecodedAudio {
        samples: make_waveform(case.samples, case.seed, case.amplitude),
        sample_rate: case.sample_rate,
    }
}

fn rows(output: &PreprocessedEncoderInputs) -> Vec<i64> {
    match output.model_specific.get(MUSE_SPARK_SAMPLES_PER_AUDIO_KEY) {
        Some(ModelSpecificValue::IntTensor { data, shape }) => {
            assert_eq!(shape, &[data.len()]);
            data.clone()
        }
        value => panic!("expected {MUSE_SPARK_SAMPLES_PER_AUDIO_KEY} IntTensor, got {value:?}"),
    }
}

#[test]
fn fixture_describes_the_family_extractor() {
    let golden = golden();
    let params = MuseSparkAudioProcessor::default().params().clone();
    assert_eq!(golden.sample_rate, params.sample_rate);
    assert_eq!(golden.samples_per_token, params.samples_per_token);
    assert!(golden.clips.len() >= 10);
}

#[test]
fn every_clip_matches_the_reference_rows_bit_for_bit() {
    let golden = golden();
    let processor = MuseSparkAudioProcessor::default();
    for case in &golden.clips {
        let output = processor
            .preprocess_decoded(&[decoded(case)])
            .unwrap_or_else(|error| panic!("{}: preprocessing failed: {error}", case.name));
        let flat = output.encoder_input.flat_f32();
        assert_eq!(
            output.feature_token_counts,
            vec![case.tokens],
            "{}: tokens",
            case.name
        );
        assert_eq!(
            output.encoder_input.shape(),
            &[case.padded_samples, 1],
            "{}: shape",
            case.name
        );
        assert_eq!(
            rows(&output),
            vec![case.padded_samples as i64],
            "{}: samples_per_audio",
            case.name
        );
        assert_eq!(
            case.padded_samples,
            case.tokens * golden.samples_per_token,
            "{}: fixture padding",
            case.name
        );
        assert!(
            flat[case.resampled_samples..]
                .iter()
                .all(|&sample| sample == 0.0),
            "{}: tail padding",
            case.name
        );
        assert_eq!(
            fnv1a_f32(&flat),
            case.fnv1a_f32,
            "{}: float32 rows",
            case.name
        );
        assert_eq!(
            fnv1a_bf16(&flat),
            case.fnv1a_bf16,
            "{}: bfloat16 rows",
            case.name
        );
    }
}

#[test]
fn batches_concatenate_the_clips_in_order() {
    let golden = golden();
    let processor = MuseSparkAudioProcessor::default();
    for batch in &golden.batches {
        let clips: Vec<DecodedAudio> = batch
            .clips
            .iter()
            .map(|name| {
                golden
                    .clips
                    .iter()
                    .find(|case| &case.name == name)
                    .map(decoded)
                    .unwrap_or_else(|| panic!("{}: unknown clip {name}", batch.name))
            })
            .collect();
        let output = processor
            .preprocess_decoded(&clips)
            .unwrap_or_else(|error| panic!("{}: preprocessing failed: {error}", batch.name));
        assert_eq!(rows(&output), batch.samples_per_audio, "{}", batch.name);
        let total: i64 = batch.samples_per_audio.iter().sum();
        assert_eq!(
            output.encoder_input.shape(),
            &[total as usize, 1],
            "{}: shape",
            batch.name
        );
        assert_eq!(
            output.item_sizes,
            batch
                .samples_per_audio
                .iter()
                .map(|&rows| (rows as u32, 1))
                .collect::<Vec<_>>(),
            "{}: item sizes",
            batch.name
        );
        let flat = output.encoder_input.flat_f32();
        assert_eq!(
            fnv1a_f32(&flat),
            batch.fnv1a_f32,
            "{}: float32 rows",
            batch.name
        );
        assert_eq!(
            fnv1a_bf16(&flat),
            batch.fnv1a_bf16,
            "{}: bfloat16 rows",
            batch.name
        );
    }
}
