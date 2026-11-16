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

use std::{
    borrow::Cow,
    f64::consts::PI,
    fmt,
    sync::{Arc, OnceLock},
};

use ndarray::Array2;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::{
    audio::{
        transforms::{bandlimited_resample, mel_bands, mel_basis},
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

/// The frames of a clip run through the FFT so many at a time: the chunk's
/// complex frames stay in cache and one FFT call covers them all.
const FRAMES_PER_CHUNK: usize = 32;

#[derive(Debug, Clone, Default)]
pub struct ParakeetAudioProcessor {
    params: ParakeetAudioParams,
    /// The tables and the FFT plan of `params`, built on the first clip and
    /// shared by every clone of the processor.
    frontend: OnceLock<Arc<LogMelFrontend>>,
}

impl ParakeetAudioProcessor {
    pub fn new(params: ParakeetAudioParams) -> Self {
        Self {
            params,
            frontend: OnceLock::new(),
        }
    }

    /// The normalized log-mel frames of a batch of decoded audios, each split
    /// into clips of at most thirty seconds. The audios are borrowed: each is
    /// copied once, into the batch's padded buffer; an audio at another rate
    /// is resampled into a buffer of its own first.
    pub fn preprocess_decoded_clips(
        &self,
        audios: &[&DecodedAudio],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if audios.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let n_mels = self.params.n_mels;
        let frontend = self
            .frontend
            .get_or_init(|| Arc::new(LogMelFrontend::build(&self.params)));
        let mut scratch = LogMelScratch::new(frontend);
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
            scratch.padded.clear();
            scratch.padded.extend_from_slice(&samples);
            scratch.padded.resize(sizes.iter().sum(), 0.0);
            let (mut start, mut frames, mut tokens) = (0, 0, 0);
            for &size in &sizes {
                let valid = size / HOP_LENGTH;
                let first_row = rows.len();
                rows.resize(first_row + valid * n_mels, 0.0);
                frontend.log_mel_into(start, size, &mut scratch, &mut rows[first_row..]);
                start += size;
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

    /// The audio's samples at the frontend's rate: borrowed when the audio
    /// is at it already, resampled otherwise.
    fn resampled<'a>(&self, decoded: &'a DecodedAudio) -> Result<Cow<'a, [f32]>, TransformError> {
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
            return Ok(Cow::Borrowed(decoded.samples.as_slice()));
        }
        Ok(Cow::Owned(bandlimited_resample(
            &decoded.samples,
            decoded.sample_rate,
            self.params.sample_rate,
        )?))
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
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let decoded: Vec<&DecodedAudio> = clips.iter().map(|clip| clip.decoded()).collect();
        self.preprocess_decoded_clips(&decoded)
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

/// What the log-mel frontend needs beside the samples, built once per
/// configuration: the centered window, the filter bank with the bin band
/// each filter's weight lies in, and the FFT plan.
struct LogMelFrontend {
    n_mels: usize,
    window: Vec<f32>,
    /// `n_mels` rows of `N_FFT / 2 + 1` weights.
    filters: Vec<f32>,
    /// Per filter, the half-open bin range outside which its weight is zero.
    bands: Vec<(usize, usize)>,
    fft: Arc<dyn Fft<f32>>,
}

impl fmt::Debug for LogMelFrontend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogMelFrontend")
            .field("n_mels", &self.n_mels)
            .field("window", &self.window.len())
            .finish()
    }
}

/// The buffers one batch reuses for every clip: the audio padded to whole
/// clips, the pre-emphasized clip with its half-frame zero padding, a chunk
/// of complex frames, the FFT's own scratch and the chunk's power spectra.
struct LogMelScratch {
    padded: Vec<f32>,
    signal: Vec<f32>,
    frames: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    power: Vec<f32>,
}

impl LogMelScratch {
    fn new(frontend: &LogMelFrontend) -> Self {
        let bins = N_FFT / 2 + 1;
        Self {
            padded: Vec::new(),
            signal: Vec::new(),
            frames: vec![Complex32::new(0.0, 0.0); FRAMES_PER_CHUNK * N_FFT],
            fft_scratch: vec![Complex32::new(0.0, 0.0); frontend.fft.get_inplace_scratch_len()],
            power: vec![0.0_f32; FRAMES_PER_CHUNK * bins],
        }
    }
}

impl LogMelFrontend {
    fn build(params: &ParakeetAudioParams) -> Self {
        let bins = N_FFT / 2 + 1;
        let filters = mel_basis(params.sample_rate, N_FFT, params.n_mels);
        let bands = mel_bands(&filters, params.n_mels, bins);
        Self {
            n_mels: params.n_mels,
            window: centered_window(),
            filters,
            bands,
            fft: FftPlanner::<f32>::new().plan_fft_forward(N_FFT),
        }
    }

    /// The normalized log-mel frames of the clip of `size` samples at `start`
    /// of the batch's padded audio into `out` (`size / hop` rows of `n_mels`),
    /// as the frame-by-frame computation produced them: the same pre-emphasis
    /// and padding, the same window, the same FFT of every frame, the same
    /// power term, the same product in the same bin order, the same
    /// standardization, so the bits do not move.
    fn log_mel_into(
        &self,
        start: usize,
        size: usize,
        scratch: &mut LogMelScratch,
        out: &mut [f32],
    ) {
        let half = N_FFT / 2;
        let bins = half + 1;
        let n_mels = self.n_mels;
        let clip = &scratch.padded[start..start + size];
        // Pre-emphasis, then the centered STFT's zero padding of half a frame
        // on each side.
        scratch.signal.clear();
        scratch.signal.resize(size + N_FFT, 0.0);
        for (t, &x) in clip.iter().enumerate() {
            scratch.signal[half + t] = if t == 0 {
                x
            } else {
                x - PREEMPHASIS * clip[t - 1]
            };
        }
        // Every frame of the clip, the ones past the valid count included, as
        // the frame-by-frame computation computed them before it kept the
        // first `valid`; only the valid ones are written.
        let n_frames = 1 + size / HOP_LENGTH;
        let valid = size / HOP_LENGTH;
        let mut features = vec![0.0_f32; n_frames * n_mels];
        let mut frame = 0;
        while frame < n_frames {
            let count = FRAMES_PER_CHUNK.min(n_frames - frame);
            for k in 0..count {
                let begin = (frame + k) * HOP_LENGTH;
                let buffer = &mut scratch.frames[k * N_FFT..(k + 1) * N_FFT];
                for (slot, (&sample, &weight)) in buffer.iter_mut().zip(
                    scratch.signal[begin..begin + N_FFT]
                        .iter()
                        .zip(&self.window),
                ) {
                    *slot = Complex32::new(sample * weight, 0.0);
                }
            }
            self.fft.process_with_scratch(
                &mut scratch.frames[..count * N_FFT],
                &mut scratch.fft_scratch,
            );
            for k in 0..count {
                let spectrum = &scratch.frames[k * N_FFT..k * N_FFT + bins];
                let power = &mut scratch.power[k * bins..(k + 1) * bins];
                for (value, c) in power.iter_mut().zip(spectrum) {
                    *value = c.re * c.re + c.im * c.im;
                }
            }
            // The mel product over each filter's band: a zero weight adds
            // nothing to a finite power, so the band's sum is the full row's
            // sum, bit for bit. Where a power overflowed, a zero weight times
            // infinity is NaN and the full row keeps it.
            for k in 0..count {
                let power = &scratch.power[k * bins..(k + 1) * bins];
                let finite = power.iter().all(|value| value.is_finite());
                let row = &mut features[(frame + k) * n_mels..(frame + k + 1) * n_mels];
                for (mel, slot) in row.iter_mut().enumerate() {
                    let (first, last) = if finite { self.bands[mel] } else { (0, bins) };
                    let weights = &self.filters[mel * bins + first..mel * bins + last];
                    let mut energy = 0.0_f32;
                    for (weight, value) in weights.iter().zip(&power[first..last]) {
                        energy += weight * value;
                    }
                    *slot = (energy + LOG_ZERO_GUARD).ln();
                }
            }
            frame += count;
        }
        out[..valid * n_mels].copy_from_slice(&normalized(&features, valid, n_mels));
    }
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
    use crate::test_support;

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
        assert!(processor.preprocess_decoded_clips(&[&audio(600)]).is_err());
        assert!(processor.preprocess_decoded_clips(&[&audio(1)]).is_ok());
        // A clip far past the cap at another rate is refused before resampling.
        let hour_at_48k = DecodedAudio {
            samples: vec![0.0; 3600 * 48_000],
            sample_rate: 48_000,
        };
        assert!(processor.preprocess_decoded_clips(&[&hour_at_48k]).is_err());
    }

    #[test]
    fn counts_tokens_like_the_encoder_subsampling() {
        // 10 s, 30 s and the 15.95 s reference clip.
        assert_eq!(parakeet_clip_tokens(1000), 125);
        assert_eq!(parakeet_clip_tokens(3000), 375);
        assert_eq!(parakeet_clip_tokens(1595), 200);
    }

    /// The computation this file replaced, kept here as the bit-identity
    /// reference: one clip at a time, the window and the plan built per call.
    fn frame_by_frame_reference(
        processor: &ParakeetAudioProcessor,
        decoded: &DecodedAudio,
    ) -> Vec<f32> {
        let n_mels = processor.params.n_mels;
        let mel_filters = mel_basis(processor.params.sample_rate, N_FFT, n_mels);
        let samples = if decoded.sample_rate == processor.params.sample_rate {
            decoded.samples.clone()
        } else {
            bandlimited_resample(
                &decoded.samples,
                decoded.sample_rate,
                processor.params.sample_rate,
            )
            .expect("resample")
        };
        let sizes = processor.params.clip_sizes(samples.len());
        let mut padded = samples;
        padded.resize(sizes.iter().sum(), 0.0);
        let mut rows = Vec::new();
        let mut start = 0;
        for &size in &sizes {
            let features =
                frame_by_frame_log_mel(&padded[start..start + size], &mel_filters, n_mels);
            start += size;
            rows.extend(normalized(&features, size / HOP_LENGTH, n_mels));
        }
        rows
    }

    fn frame_by_frame_log_mel(clip: &[f32], mel_filters: &[f32], n_mels: usize) -> Vec<f32> {
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

    /// Audios of the lengths that exercise the clip and frame geometry: under
    /// the shortest tail, a second, either side of the 32-frame chunk
    /// boundary (31 and 33 frames at a 160 hop), a whole clip, a clip and a
    /// tail, two clips and a tail; quiet, ordinary and loud.
    fn waveform_corpus() -> Vec<(String, DecodedAudio)> {
        test_support::waveform_corpus(&[
            1, 1600, 4800, 4960, 5120, 16_000, 480_000, 480_800, 961_600,
        ])
    }

    fn rows_of(processor: &ParakeetAudioProcessor, audios: &[&DecodedAudio]) -> Vec<f32> {
        let inputs = processor
            .preprocess_decoded_clips(audios)
            .expect("audio preprocesses");
        let values = inputs.encoder_input.as_f32();
        values.as_slice().expect("contiguous").to_vec()
    }

    fn assert_same_bits(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{name}: length");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.to_bits() == e.to_bits(),
                "{name}: value {index} is {a} ({:#x}), the frame-by-frame computation gives {e} ({:#x})",
                a.to_bits(),
                e.to_bits()
            );
        }
    }

    #[test]
    fn the_batched_frontend_equals_the_frame_by_frame_computation_bit_for_bit() {
        let processor = ParakeetAudioProcessor::default();
        for (name, decoded) in waveform_corpus() {
            let actual = rows_of(&processor, &[&decoded]);
            let expected = frame_by_frame_reference(&processor, &decoded);
            assert_same_bits(&name, &actual, &expected);
        }
    }

    #[test]
    fn a_batch_equals_each_audio_alone_in_order() {
        let processor = ParakeetAudioProcessor::default();
        let corpus = waveform_corpus();
        let batch: Vec<&DecodedAudio> = corpus.iter().map(|(_, clip)| clip).collect();
        let rows = rows_of(&processor, &batch);
        let mut offset = 0;
        for (name, clip) in &corpus {
            let alone = rows_of(&processor, &[clip]);
            assert_eq!(
                &rows[offset..offset + alone.len()],
                &alone[..],
                "{name}: rows"
            );
            offset += alone.len();
        }
        assert_eq!(offset, rows.len());
    }

    #[test]
    fn a_spectrum_that_overflows_takes_the_full_product_and_keeps_its_bits() {
        let processor = ParakeetAudioProcessor::default();
        let decoded = DecodedAudio {
            samples: (0..4_000)
                .map(|i| if i % 2 == 0 { 1e30 } else { -1e30 })
                .collect(),
            sample_rate: 16_000,
        };
        let actual = rows_of(&processor, &[&decoded]);
        let expected = frame_by_frame_reference(&processor, &decoded);
        assert_same_bits("overflowing clip", &actual, &expected);
    }

    #[test]
    fn a_processor_clone_shares_the_built_frontend() {
        let processor = ParakeetAudioProcessor::default();
        let clip = DecodedAudio {
            samples: vec![0.1; 16_000],
            sample_rate: 16_000,
        };
        rows_of(&processor, &[&clip]);
        let clone = processor.clone();
        assert!(Arc::ptr_eq(
            processor.frontend.get().expect("built"),
            clone.frontend.get().expect("shared")
        ));
    }
}
