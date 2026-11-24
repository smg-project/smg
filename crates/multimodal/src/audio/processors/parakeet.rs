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
        transforms::{bandlimited_resample, mel_bands, mel_basis, resampled_len},
        AudioPreProcessor, DecodedAudio,
    },
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
};

const N_FFT: usize = 512;
const BINS: usize = N_FFT / 2 + 1;
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
    pub(crate) fn from_model_config(
        config: &serde_json::Value,
    ) -> Result<Self, ParakeetConfigError> {
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

    fn clip_layout(&self, len: usize) -> (usize, usize, Option<usize>) {
        let target = (CLIP_SECONDS * self.sample_rate as f64).round() as usize;
        let tail_min = (TAIL_MIN_SECONDS * self.sample_rate as f64).round() as usize;
        let len = len.max(tail_min);
        let tail = (!len.is_multiple_of(target)).then(|| (len % target).max(tail_min));
        (target, len / target, tail)
    }

    /// The sample counts of the clips an audio of `len` samples splits into.
    #[cfg(test)]
    fn clip_sizes(&self, len: usize) -> Vec<usize> {
        let (target, full, tail) = self.clip_layout(len);
        let mut sizes = vec![target; full];
        if let Some(tail) = tail {
            sizes.push(tail);
        }
        sizes
    }
}

/// Encoder positions a clip of `frames` mel frames becomes after subsampling.
pub(crate) fn parakeet_clip_tokens(frames: usize) -> usize {
    (0..SUBSAMPLING_STAGES).fold(frames, |n, _| if n == 0 { 0 } else { (n - 1) / 2 + 1 })
}

/// One 64-byte cache line of `f32` per bin: keeps `power` and a chunk's output rows inside 32 KiB L1d.
const FRAMES_PER_CHUNK: usize = 16;
const CHUNK_SIGNAL_LEN: usize = (FRAMES_PER_CHUNK - 1) * HOP_LENGTH + N_FFT;

#[derive(Clone, Copy)]
#[repr(align(64))]
struct ChunkFrames([f32; FRAMES_PER_CHUNK]);

#[derive(Debug, Clone, Default)]
pub struct ParakeetAudioProcessor {
    params: ParakeetAudioParams,
    /// The tables and the FFT plan of `params`, built on the first clip and
    /// shared by every clone of the processor.
    frontend: OnceLock<Arc<LogMelFrontend>>,
}

impl ParakeetAudioProcessor {
    pub(crate) fn new(params: ParakeetAudioParams) -> Self {
        Self {
            params,
            frontend: OnceLock::new(),
        }
    }

    /// Standardized log-mel frames and encoder token counts for `audios`.
    pub fn preprocess_decoded_clips(
        &self,
        audios: &[&DecodedAudio],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        TransformError::if_empty(audios)?;
        let n_mels = self.params.n_mels;
        if self.params.sample_rate == 0 || n_mels == 0 {
            return Err(TransformError::ShapeError(
                "Parakeet sample_rate and n_mels must be positive".to_string(),
            ));
        }
        let max_samples = MAX_AUDIO_SECONDS.saturating_mul(self.params.sample_rate);
        let overflow = || TransformError::ShapeError("Parakeet audio size overflow".to_string());
        let (mut total_clips, mut total_frames) = (0usize, 0usize);
        for audio in audios {
            if audio.sample_rate == 0 {
                return Err(TransformError::ShapeError(
                    "decoded audio sample rate must be positive".to_string(),
                ));
            }
            if audio.samples.len() > (MAX_AUDIO_SECONDS + 1).saturating_mul(audio.sample_rate) {
                return Err(too_long());
            }
            let len = resampled_len(
                audio.samples.len(),
                audio.sample_rate,
                self.params.sample_rate,
            )?;
            if len > max_samples {
                return Err(too_long());
            }
            let (target, full, tail) = self.params.clip_layout(len);
            let clips = full + usize::from(tail.is_some());
            let frames = full
                .checked_mul(target / HOP_LENGTH)
                .and_then(|n| n.checked_add(tail.map_or(0, |t| t / HOP_LENGTH)))
                .ok_or_else(overflow)?;
            total_clips = total_clips.checked_add(clips).ok_or_else(overflow)?;
            total_frames = total_frames.checked_add(frames).ok_or_else(overflow)?;
        }
        let total_elems = total_frames.checked_mul(n_mels).ok_or_else(overflow)?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(total_elems).map_err(|error| {
            TransformError::ShapeError(format!(
                "failed to reserve {total_elems} Parakeet audio features: {error}"
            ))
        })?;
        rows.resize(total_elems, 0.0);

        let frontend = self
            .frontend
            .get_or_init(|| Arc::new(LogMelFrontend::build(&self.params)));
        let mut scratch = LogMelScratch::new(frontend, n_mels);
        let mut token_counts = Vec::with_capacity(audios.len());
        let mut item_sizes = Vec::with_capacity(audios.len());
        let mut clip_frames = Vec::with_capacity(total_clips);
        let mut clips_per_item = Vec::with_capacity(audios.len());
        let mut frames_per_item = Vec::with_capacity(audios.len());
        let mut row_cursor = 0usize;
        for audio in audios {
            let samples = self.resampled(audio)?;
            if samples.len() > max_samples {
                return Err(too_long());
            }
            let (target, full, tail) = self.params.clip_layout(samples.len());
            let clips = full + usize::from(tail.is_some());
            let (n, mut start, mut frames, mut tokens) = (samples.len(), 0, 0, 0);
            for size in std::iter::repeat_n(target, full).chain(tail) {
                let actual = &samples[start.min(n)..start.saturating_add(size).min(n)];
                let valid = size / HOP_LENGTH;
                let next_row = valid
                    .checked_mul(n_mels)
                    .and_then(|count| row_cursor.checked_add(count))
                    .ok_or_else(overflow)?;
                let slice = rows.get_mut(row_cursor..next_row).ok_or_else(|| {
                    TransformError::ShapeError("resampled audio exceeds reserved rows".to_string())
                })?;
                frontend.log_mel_into(actual, size, &mut scratch, slice);
                start += size;
                row_cursor = next_row;
                clip_frames.push(valid as i64);
                frames += valid;
                tokens += parakeet_clip_tokens(valid);
            }
            token_counts.push(tokens);
            item_sizes.push((n_mels as u32, frames as u32));
            clips_per_item.push(clips as i64);
            frames_per_item.push(frames as i64);
        }
        if row_cursor != total_elems {
            return Err(TransformError::ShapeError(
                "resampled audio length differs from reserved rows".to_string(),
            ));
        }

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
        if decoded.samples.len() > (MAX_AUDIO_SECONDS + 1).saturating_mul(decoded.sample_rate) {
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

struct LogMelScratch {
    signal: [f32; CHUNK_SIGNAL_LEN],
    power: [ChunkFrames; BINS],
    frames: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    sums: Vec<f64>,
    vars: Vec<f64>,
}

impl LogMelScratch {
    fn new(frontend: &LogMelFrontend, n_mels: usize) -> Self {
        Self {
            signal: [0.0_f32; CHUNK_SIGNAL_LEN],
            power: [ChunkFrames([0.0_f32; FRAMES_PER_CHUNK]); BINS],
            frames: vec![Complex32::new(0.0, 0.0); FRAMES_PER_CHUNK * N_FFT],
            fft_scratch: vec![Complex32::new(0.0, 0.0); frontend.fft.get_inplace_scratch_len()],
            sums: vec![-0.0; n_mels],
            vars: vec![-0.0; n_mels],
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

    /// Fill `out` (`[size / HOP_LENGTH, n_mels]`), zero-padding `actual` to `size` samples.
    fn log_mel_into(
        &self,
        actual: &[f32],
        size: usize,
        scratch: &mut LogMelScratch,
        out: &mut [f32],
    ) {
        let n_mels = self.n_mels;
        let valid = size / HOP_LENGTH;
        if valid == 0 || n_mels == 0 {
            return;
        }
        scratch.sums.fill(-0.0);
        scratch.vars.fill(-0.0);

        let mut frame = 0;
        while frame < valid {
            let count = FRAMES_PER_CHUNK.min(valid - frame);
            fill_chunk_frames(
                actual,
                size,
                &self.window,
                frame,
                &mut scratch.signal,
                &mut scratch.frames[..count * N_FFT],
            );
            self.fft.process_with_scratch(
                &mut scratch.frames[..count * N_FFT],
                &mut scratch.fft_scratch,
            );
            power_and_mel_chunk(
                &scratch.frames[..count * N_FFT],
                &mut scratch.power,
                &self.filters,
                &self.bands,
                &mut out[frame * n_mels..(frame + count) * n_mels],
                &mut scratch.sums,
            );
            frame += count;
        }
        finish_normalize_in_place(out, &mut scratch.sums, &mut scratch.vars);
    }
}

#[cfg_attr(
    target_arch = "x86_64",
    multiversion::multiversion(targets("x86_64+avx2+fma"))
)]
fn fill_chunk_frames(
    actual: &[f32],
    size: usize,
    window: &[f32],
    frame: usize,
    chunk_signal: &mut [f32; CHUNK_SIGNAL_LEN],
    frames: &mut [Complex32],
) {
    let (frame_chunks, rem) = frames.as_chunks_mut::<N_FFT>();
    let count = frame_chunks.len();
    debug_assert!(rem.is_empty() && (1..=FRAMES_PER_CHUNK).contains(&count));
    let (Some(window), 1..=FRAMES_PER_CHUNK) = (window.first_chunk::<N_FFT>(), count) else {
        return;
    };
    let (half, base) = (N_FFT / 2, frame * HOP_LENGTH);
    let chunk_len = (count - 1) * HOP_LENGTH + N_FFT;
    let signal = &mut chunk_signal[..chunk_len];
    let lead = half.saturating_sub(base).min(chunk_len);
    let t0 = (base + lead) - half;
    let t1 = (base + chunk_len) - half;
    let n = actual.len();
    if n > 0 && t0 < t1 {
        signal[..lead].fill(0.0);
        let mut written = lead;
        if t0 == 0 {
            signal[written] = actual[0];
            written += 1;
        }
        let (a, b) = (1.max(t0), n.min(t1));
        if a < b {
            let span = b - a;
            for (slot, (&x, &prev)) in signal[written..written + span]
                .iter_mut()
                .zip(actual[a..b].iter().zip(&actual[a - 1..b - 1]))
            {
                *slot = x - PREEMPHASIS * prev;
            }
            written += span;
        }
        if n < size && t0 <= n && n < t1 {
            // Tail padding runs before pre-emphasis, while STFT padding runs after.
            signal[written] = 0.0 - PREEMPHASIS * actual[n - 1];
            written += 1;
        }
        signal[written..].fill(0.0);
    } else {
        signal.fill(0.0);
    }

    for (k, dst) in frame_chunks.iter_mut().enumerate() {
        let src = &signal[k * HOP_LENGTH..k * HOP_LENGTH + N_FFT];
        for (slot, (&sample, &weight)) in dst.iter_mut().zip(src.iter().zip(window)) {
            *slot = Complex32::new(sample * weight, 0.0);
        }
    }
}

#[cfg_attr(
    target_arch = "x86_64",
    multiversion::multiversion(targets("x86_64+avx2+fma"))
)]
fn power_and_mel_chunk(
    frames: &[Complex32],
    power: &mut [ChunkFrames; BINS],
    filters: &[f32],
    bands: &[(usize, usize)],
    out_chunk: &mut [f32],
    sums: &mut [f64],
) {
    let n_mels = bands.len();
    let (frame_chunks, rem) = frames.as_chunks::<N_FFT>();
    let (filter_rows, filter_rem) = filters.as_chunks::<BINS>();
    let count = frame_chunks.len();
    debug_assert!(rem.is_empty() && filter_rem.is_empty() && filter_rows.len() == n_mels);
    debug_assert!(count <= FRAMES_PER_CHUNK && out_chunk.len() == count * n_mels);
    let Some(sums) = sums
        .get_mut(..n_mels)
        .filter(|_| n_mels > 0 && count <= FRAMES_PER_CHUNK)
    else {
        return;
    };
    let mut chunk_finite = true;
    let mut row_power = [0.0_f32; BINS];
    for (k, frame) in frame_chunks.iter().enumerate() {
        for (slot, c) in row_power.iter_mut().zip(&frame[..BINS]) {
            *slot = c.re * c.re + c.im * c.im;
        }
        chunk_finite &= row_power.iter().fold(true, |ok, p| ok & p.is_finite());
        for (bin_lanes, &p) in power.iter_mut().zip(&row_power) {
            bin_lanes.0[k] = p;
        }
    }
    if count < FRAMES_PER_CHUNK {
        for bin_lanes in power.iter_mut() {
            bin_lanes.0[count..].fill(0.0);
        }
    }
    for (mel, (&band, filter_row)) in bands.iter().zip(filter_rows).enumerate() {
        // Full-row fallback keeps `0.0 * inf = NaN` without changing finite lanes.
        let (first, last) = if chunk_finite { band } else { (0, BINS) };
        let mut energy = ChunkFrames([0.0_f32; FRAMES_PER_CHUNK]);
        for (&weight, bin_power) in filter_row[first..last].iter().zip(&power[first..last]) {
            for (acc, &p) in energy.0.iter_mut().zip(&bin_power.0) {
                *acc += weight * p;
            }
        }
        for (row, &e) in out_chunk.chunks_exact_mut(n_mels).zip(&energy.0) {
            row[mel] = (e + LOG_ZERO_GUARD).ln();
        }
    }
    for row in out_chunk.chunks_exact(n_mels) {
        for (sum, &x) in sums.iter_mut().zip(row) {
            *sum += f64::from(x);
        }
    }
}

/// Summing in `f64` in frame order stops `f32` drift on quiet bins and keeps bit-identity.
#[cfg_attr(
    target_arch = "x86_64",
    multiversion::multiversion(targets("x86_64+avx2+fma"))
)]
fn finish_normalize_in_place(rows: &mut [f32], sums: &mut [f64], vars: &mut [f64]) {
    let n_mels = sums.len();
    debug_assert!(vars.len() == n_mels && n_mels > 0 && rows.len().is_multiple_of(n_mels));
    let (Some(vars), true) = (vars.get_mut(..n_mels), n_mels > 0) else {
        return;
    };
    let count = (rows.len() / n_mels) as f64;
    for sum in sums.iter_mut() {
        *sum /= count;
    }
    for row in rows.chunks_exact(n_mels) {
        for ((var, &x), &mean) in vars.iter_mut().zip(row).zip(&*sums) {
            let diff = f64::from(x) - mean;
            *var += diff * diff;
        }
    }
    let denom = count - 1.0;
    for var in vars.iter_mut() {
        *var = (*var / denom).sqrt() + f64::from(NORM_EPSILON);
    }
    for row in rows.chunks_exact_mut(n_mels) {
        for ((slot, &mean), &scale) in row.iter_mut().zip(&*sums).zip(&*vars) {
            *slot = ((f64::from(*slot) - mean) / scale) as f32;
        }
    }
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

    /// Sample counts spanning sub-tail padding, chunk boundaries, and multi-clip splits.
    fn waveform_corpus() -> Vec<(String, DecodedAudio)> {
        test_support::waveform_corpus(&[
            1, 1600, 2400, 2560, 2720, 4800, 4960, 5120, 16_000, 480_000, 480_800, 961_600,
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
        let mut corpus = waveform_corpus();
        for (rate, len) in [(8_000, 1_001), (44_100, 4_411), (48_000, 4_801)] {
            corpus.push((
                format!("resampled_{rate}_{len}"),
                DecodedAudio {
                    samples: (0..len).map(|i| (i as f32 * 0.02).sin() * 0.4).collect(),
                    sample_rate: rate,
                },
            ));
        }
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

        let mut samples: Vec<f32> = (0..6_400).map(|i| (i as f32 * 0.03).sin() * 0.25).collect();
        samples[..160].copy_from_slice(&decoded.samples[..160]);
        let mixed = DecodedAudio {
            samples,
            sample_rate: 16_000,
        };
        let actual = rows_of(&processor, &[&mixed]);
        let expected = frame_by_frame_reference(&processor, &mixed);
        assert_same_bits("mixed chunk", &actual, &expected);

        let fe = LogMelFrontend::build(&processor.params);
        let n_mels = processor.params.n_mels;
        let mut sc = LogMelScratch::new(&fe, n_mels);
        let mut raw = vec![0.0_f32; FRAMES_PER_CHUNK * n_mels];
        let s = &mixed.samples;
        fill_chunk_frames(s, s.len(), &fe.window, 0, &mut sc.signal, &mut sc.frames);
        fe.fft
            .process_with_scratch(&mut sc.frames, &mut sc.fft_scratch);
        power_and_mel_chunk(
            &sc.frames,
            &mut sc.power,
            &fe.filters,
            &fe.bands,
            &mut raw,
            &mut sc.sums,
        );
        let expected_raw = frame_by_frame_log_mel(s, &fe.filters, n_mels);
        assert!(raw[..n_mels].iter().any(|v| v.is_nan()));
        assert!(raw[3 * n_mels..].iter().all(|v| v.is_finite()));
        assert_same_bits("mixed raw", &raw, &expected_raw[..raw.len()]);
    }

    #[test]
    fn non_default_geometries_and_chunk_boundary_tails_keep_their_bits() {
        let build = |sample_rate, n_mels| {
            ParakeetAudioProcessor::new(ParakeetAudioParams {
                sample_rate,
                n_mels,
            })
        };
        let s = FRAMES_PER_CHUNK * HOP_LENGTH - N_FFT / 2;
        let e = (FRAMES_PER_CHUNK - 1) * HOP_LENGTH + N_FFT / 2;
        let tails = [s - 1, s, s + 1, e - 1, e, e + 1].map(|len| (64_000, 128, len));
        for &(sr, n_mels, len) in [(16_000, 80, 16_800), (16_000, 131, 16_800)]
            .iter()
            .chain(&tails)
        {
            let processor = build(sr, n_mels);
            let decoded = DecodedAudio {
                samples: (0..len).map(|i| (i as f32 * 0.013).cos() * 0.35).collect(),
                sample_rate: sr,
            };
            let actual = rows_of(&processor, &[&decoded]);
            let expected = frame_by_frame_reference(&processor, &decoded);
            assert_same_bits(&format!("{sr}/{n_mels}/{len}"), &actual, &expected);
        }
        let clip = DecodedAudio {
            samples: vec![0.1; 1600],
            sample_rate: 16_000,
        };
        for (sr, n_mels) in [(0, 128), (16_000, 0)] {
            assert!(build(sr, n_mels)
                .preprocess_decoded_clips(&[&clip])
                .is_err());
        }
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
