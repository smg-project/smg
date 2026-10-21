//! Parakeet (Nemotron-H Omni) audio preprocessing.
//!
//! Port of vLLM's `ParakeetExtractor`. Audio is resampled to the configured
//! rate and split into 30-second clips (a short tail is zero-padded to 0.1 s);
//! each clip is pre-emphasized, framed by a centered, zero-padded STFT (512
//! bins, hop 160, a 400-sample symmetric Hann window centered in each frame),
//! mapped to Slaney mel power, logged with a `2^-24` guard and normalized per
//! mel bin over its `samples / hop` valid frames (unbiased deviation). The
//! engine receives each clip's valid frames as `(frames, n_mels)` rows with
//! `audio_clip_frames`, and its encoder subsamples every clip 8x. Audio longer
//! than vLLM's default decode limit of 600 s is refused, as vLLM refuses it.

use std::f64::consts::PI;

use ndarray::Array2;
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::{
    audio::{
        transforms::{bandlimited_resample, mel_basis},
        AudioPreProcessor, DecodedAudio,
    },
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
};

const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const PREEMPHASIS: f32 = 0.97;
const CLIP_SECONDS: f64 = 30.0;
/// vLLM's default `VLLM_MAX_AUDIO_DECODE_DURATION_S`.
const MAX_AUDIO_SECONDS: usize = 600;
const TAIL_MIN_SECONDS: f64 = 0.1;
const LOG_ZERO_GUARD: f32 = 1.0 / (1u32 << 24) as f32;
const NORM_EPSILON: f32 = 1e-5;
/// Stride-2 convolutions in the encoder's 8x subsampling.
const SUBSAMPLING_STAGES: usize = 3;

/// The `sound_config` field that keeps a model config off this port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParakeetConfigError {
    /// A field the port reads is absent or not a count.
    Missing(&'static str),
    /// A field differs from the value the port implements, or is a zero count.
    Unsupported(&'static str),
}

#[derive(Debug, Clone)]
pub struct ParakeetAudioParams {
    pub sample_rate: usize,
    pub n_mels: usize,
}

impl Default for ParakeetAudioParams {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_mels: 128,
        }
    }
}

impl ParakeetAudioParams {
    /// Read `sound_config.{sampling_rate,num_mel_bins}` from the model config,
    /// refusing an encoder that does not subsample 8x with stride-2 kernel-3
    /// convolutions or a config that overrides the extractor settings this
    /// port implements.
    pub fn from_model_config(config: &serde_json::Value) -> Result<Self, ParakeetConfigError> {
        let sound = config
            .get("sound_config")
            .ok_or(ParakeetConfigError::Missing("sound_config"))?;
        let count = |name: &'static str| {
            let value = sound
                .get(name)
                .and_then(|v| v.as_u64())
                .and_then(|v| usize::try_from(v).ok())
                .ok_or(ParakeetConfigError::Missing(name))?;
            if value == 0 {
                return Err(ParakeetConfigError::Unsupported(name));
            }
            Ok(value)
        };
        if count("subsampling_factor")? != 1 << SUBSAMPLING_STAGES {
            return Err(ParakeetConfigError::Unsupported("subsampling_factor"));
        }
        let fixed = [
            ("subsampling_conv_stride", 2.0),
            ("subsampling_conv_kernel_size", 3.0),
            ("n_fft", N_FFT as f64),
            ("hop_length", HOP_LENGTH as f64),
            ("win_length", WIN_LENGTH as f64),
            ("preemphasis", 0.97),
            ("padding_value", 0.0),
        ];
        if let Some((name, _)) = fixed
            .iter()
            .find(|(name, value)| sound.get(*name).is_some_and(|v| v.as_f64() != Some(*value)))
        {
            return Err(ParakeetConfigError::Unsupported(name));
        }
        Ok(Self {
            sample_rate: count("sampling_rate")?,
            n_mels: count("num_mel_bins")?,
        })
    }

    /// The sample counts of the clips an audio of `len` samples splits into.
    pub fn clip_sizes(&self, len: usize) -> Vec<usize> {
        let target = (CLIP_SECONDS * self.sample_rate as f64).round() as usize;
        let tail_min = (TAIL_MIN_SECONDS * self.sample_rate as f64).round() as usize;
        let len = len.max(tail_min);
        let mut sizes = vec![target; len / target];
        if !len.is_multiple_of(target) {
            sizes.push((len % target).max(tail_min));
        }
        sizes
    }
}

/// Encoder positions a clip of `frames` mel frames becomes after subsampling.
pub fn parakeet_clip_tokens(frames: usize) -> usize {
    (0..SUBSAMPLING_STAGES).fold(frames, |n, _| if n == 0 { 0 } else { (n - 1) / 2 + 1 })
}

#[derive(Debug, Clone, Default)]
pub struct ParakeetAudioProcessor {
    params: ParakeetAudioParams,
}

impl ParakeetAudioProcessor {
    pub fn new(params: ParakeetAudioParams) -> Self {
        Self { params }
    }

    pub fn preprocess_decoded_clips(
        &self,
        audios: Vec<DecodedAudio>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if audios.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let n_mels = self.params.n_mels;
        let mel_filters = mel_basis(self.params.sample_rate, N_FFT, n_mels);
        let mut rows = Vec::new();
        let mut token_counts = Vec::with_capacity(audios.len());
        let mut item_sizes = Vec::with_capacity(audios.len());
        let mut clip_frames = Vec::new();
        let mut clips_per_item = Vec::with_capacity(audios.len());
        let mut frames_per_item = Vec::with_capacity(audios.len());
        for audio in audios {
            let samples = self.resampled(audio)?;
            if samples.len() > MAX_AUDIO_SECONDS * self.params.sample_rate {
                return Err(too_long());
            }
            let sizes = self.params.clip_sizes(samples.len());
            let mut padded = samples;
            padded.resize(sizes.iter().sum(), 0.0);
            let (mut start, mut frames, mut tokens) = (0, 0, 0);
            for &size in &sizes {
                let features = log_mel(&padded[start..start + size], &mel_filters, n_mels);
                start += size;
                let valid = size / HOP_LENGTH;
                rows.extend(normalized(&features, valid, n_mels));
                clip_frames.push(valid as i64);
                frames += valid;
                tokens += parakeet_clip_tokens(valid);
            }
            token_counts.push(tokens);
            item_sizes.push((n_mels as u32, frames as u32));
            clips_per_item.push(sizes.len() as i64);
            frames_per_item.push(frames as i64);
        }
        let total_frames = rows.len() / n_mels;
        let encoder_input = Array2::from_shape_vec((total_frames, n_mels), rows).map_err(|e| {
            TransformError::ShapeError(format!(
                "failed to create Parakeet audio encoder input [{total_frames}, {n_mels}]: {e}"
            ))
        })?;
        Ok(
            PreprocessedEncoderInputs::new(encoder_input, token_counts, item_sizes)
                .with_extra("audio_clip_frames", ModelSpecificValue::int_1d(clip_frames))
                .with_extra("clips_per_item", ModelSpecificValue::int_1d(clips_per_item))
                .with_extra(
                    "frames_per_item",
                    ModelSpecificValue::int_1d(frames_per_item),
                ),
        )
    }

    fn resampled(&self, decoded: DecodedAudio) -> Result<Vec<f32>, TransformError> {
        if decoded.sample_rate == 0 {
            return Err(TransformError::ShapeError(
                "decoded audio sample rate must be positive".to_string(),
            ));
        }
        // A clip well past the cap is refused before it is resampled; the cap
        // itself applies to the resampled samples, as in the ref engine.
        if decoded.samples.len() > (MAX_AUDIO_SECONDS + 1) * decoded.sample_rate {
            return Err(too_long());
        }
        if decoded.sample_rate == self.params.sample_rate {
            return Ok(decoded.samples);
        }
        bandlimited_resample(
            &decoded.samples,
            decoded.sample_rate,
            self.params.sample_rate,
        )
    }
}

fn too_long() -> TransformError {
    TransformError::ShapeError(format!(
        "audio exceeds the maximum duration of {MAX_AUDIO_SECONDS} s"
    ))
}

impl AudioPreProcessor for ParakeetAudioProcessor {
    fn preprocess(
        &self,
        clips: &[std::sync::Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_decoded_clips(clips.iter().map(|clip| clip.decoded().clone()).collect())
    }
}

/// `torch.hann_window(WIN_LENGTH, periodic=False)` centered in an `N_FFT` frame, as `torch.stft` pads it.
fn centered_window() -> Vec<f32> {
    let mut window = vec![0.0_f32; N_FFT];
    let offset = (N_FFT - WIN_LENGTH) / 2;
    for i in 0..WIN_LENGTH {
        let phase = 2.0 * PI * i as f64 / (WIN_LENGTH - 1) as f64;
        window[offset + i] = (0.5 - 0.5 * phase.cos()) as f32;
    }
    window
}

/// Log-mel frames `[1 + len / hop][n_mels]` of one pre-emphasized clip.
fn log_mel(clip: &[f32], mel_filters: &[f32], n_mels: usize) -> Vec<f32> {
    let half = N_FFT / 2;
    // Pre-emphasis, then the centered STFT's zero padding of half a frame on each side.
    let mut signal = vec![0.0_f32; clip.len() + N_FFT];
    for (t, &x) in clip.iter().enumerate() {
        signal[half + t] = if t == 0 {
            x
        } else {
            x - PREEMPHASIS * clip[t - 1]
        };
    }
    let window = centered_window();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let bins = half + 1;
    let n_frames = 1 + clip.len() / HOP_LENGTH;
    let mut buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
    let mut power = vec![0.0_f32; bins];
    let mut out = vec![0.0_f32; n_frames * n_mels];
    for frame in 0..n_frames {
        let start = frame * HOP_LENGTH;
        for (b, (&s, &w)) in buffer
            .iter_mut()
            .zip(signal[start..start + N_FFT].iter().zip(&window))
        {
            *b = Complex32::new(s * w, 0.0);
        }
        fft.process(&mut buffer);
        for (p, c) in power.iter_mut().zip(&buffer[..bins]) {
            *p = c.re * c.re + c.im * c.im;
        }
        for mel in 0..n_mels {
            let filter = &mel_filters[mel * bins..][..bins];
            let energy: f32 = filter.iter().zip(&power).map(|(f, p)| f * p).sum();
            out[frame * n_mels + mel] = (energy + LOG_ZERO_GUARD).ln();
        }
    }
    out
}

/// The first `valid` frames, each mel bin standardized over them. The statistics
/// accumulate in f64: a sequential f32 sum over a 30-second clip drifts by more
/// than a quiet bin's spread.
fn normalized(features: &[f32], valid: usize, n_mels: usize) -> Vec<f32> {
    let mut out = features[..valid * n_mels].to_vec();
    for mel in 0..n_mels {
        let values = || (0..valid).map(|t| f64::from(features[t * n_mels + mel]));
        let mean = values().sum::<f64>() / valid as f64;
        let variance =
            values().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (valid as f64 - 1.0);
        let scale = variance.sqrt() + f64::from(NORM_EPSILON);
        for t in 0..valid {
            let x = f64::from(features[t * n_mels + mel]);
            out[t * n_mels + mel] = ((x - mean) / scale) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_long_audio_into_30_second_clips_and_pads_a_short_tail() {
        let params = ParakeetAudioParams::default();
        assert_eq!(
            params.clip_sizes(480_000 * 2 + 800),
            vec![480_000, 480_000, 1600]
        );
        assert_eq!(params.clip_sizes(16_000), vec![16_000]);
        assert_eq!(params.clip_sizes(100), vec![1600]);
    }

    #[test]
    fn reads_its_parameters_from_the_sound_config() {
        let config = |sound: serde_json::Value| serde_json::json!({ "sound_config": sound });
        let params = ParakeetAudioParams::from_model_config(&config(serde_json::json!({
            "sampling_rate": 16000, "num_mel_bins": 128, "subsampling_factor": 8
        })))
        .unwrap();
        assert_eq!((params.sample_rate, params.n_mels), (16_000, 128));
        for (sound, error) in [
            (
                serde_json::json!({"sampling_rate": 16000, "subsampling_factor": 8}),
                ParakeetConfigError::Missing("num_mel_bins"),
            ),
            (
                serde_json::json!({"sampling_rate": 16000, "num_mel_bins": 128, "subsampling_factor": 4}),
                ParakeetConfigError::Unsupported("subsampling_factor"),
            ),
            (
                serde_json::json!({"sampling_rate": 16000, "num_mel_bins": 0, "subsampling_factor": 8}),
                ParakeetConfigError::Unsupported("num_mel_bins"),
            ),
            (
                serde_json::json!({"sampling_rate": 0, "num_mel_bins": 128, "subsampling_factor": 8}),
                ParakeetConfigError::Unsupported("sampling_rate"),
            ),
            (
                serde_json::json!({
                    "sampling_rate": 16000, "num_mel_bins": 128, "subsampling_factor": 8, "hop_length": 320
                }),
                ParakeetConfigError::Unsupported("hop_length"),
            ),
        ] {
            assert_eq!(
                ParakeetAudioParams::from_model_config(&config(sound)).unwrap_err(),
                error
            );
        }
    }

    #[test]
    fn refuses_audio_past_the_decode_limit() {
        let processor = ParakeetAudioProcessor::default();
        let audio = |seconds: usize| DecodedAudio {
            samples: vec![0.0; seconds * 16_000 + 1],
            sample_rate: 16_000,
        };
        // The ref engine's 600 s decode cap.
        assert!(processor
            .preprocess_decoded_clips(vec![audio(600)])
            .is_err());
        assert!(processor.preprocess_decoded_clips(vec![audio(1)]).is_ok());
        // A clip far past the cap at another rate is refused before resampling.
        let hour_at_48k = DecodedAudio {
            samples: vec![0.0; 3600 * 48_000],
            sample_rate: 48_000,
        };
        assert!(processor
            .preprocess_decoded_clips(vec![hour_at_48k])
            .is_err());
    }

    #[test]
    fn counts_tokens_like_the_encoder_subsampling() {
        // 10 s, 30 s and the 15.95 s reference clip.
        assert_eq!(parakeet_clip_tokens(1000), 125);
        assert_eq!(parakeet_clip_tokens(3000), 375);
        assert_eq!(parakeet_clip_tokens(1595), 200);
    }
}
