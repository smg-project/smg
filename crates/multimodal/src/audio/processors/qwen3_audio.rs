//! Qwen3 audio preprocessing using a Whisper-compatible log-mel frontend.

use std::{
    borrow::Cow,
    fmt,
    sync::{Arc, OnceLock},
};

use ndarray::{Array2, Array3};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use serde_json::Value;

use crate::{
    audio::{
        transforms::{bandlimited_resample, hann_window, mel_bands, mel_basis},
        AudioPreProcessor, DecodedAudio,
    },
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
    vision::PreProcessorConfig,
};

/// Parameters used by the Qwen3 audio frontend.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3AudioParams {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_window: usize,
    pub padding_value: f32,
    pub max_samples: Option<usize>,
}

impl Default for Qwen3AudioParams {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_mels: 128,
            n_fft: 400,
            hop_length: 160,
            n_window: 50,
            padding_value: 0.0,
            max_samples: None,
        }
    }
}

impl Qwen3AudioParams {
    pub(crate) fn from_configs(
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> Self {
        let mut params = Self::default();

        if let Some(value) = preprocessor_config.get_extra::<usize>("sampling_rate") {
            params.sample_rate = value;
        }
        if let Some(value) = preprocessor_config.get_extra::<usize>("feature_size") {
            params.n_mels = value;
        } else if let Some(value) = find_model_usize(
            model_config,
            &[
                &["thinker_config", "audio_config", "num_mel_bins"],
                &["audio_config", "num_mel_bins"],
            ],
        ) {
            params.n_mels = value;
        }
        if let Some(value) = preprocessor_config.get_extra::<usize>("n_fft") {
            params.n_fft = value;
        }
        if let Some(value) = preprocessor_config.get_extra::<usize>("hop_length") {
            params.hop_length = value;
        }
        if let Some(value) = preprocessor_config
            .get_extra::<usize>("n_window")
            .or_else(|| {
                find_model_usize(
                    model_config,
                    &[
                        &["thinker_config", "audio_config", "n_window"],
                        &["audio_config", "n_window"],
                    ],
                )
            })
        {
            params.n_window = value;
        }
        if let Some(value) = preprocessor_config.get_extra::<f32>("padding_value") {
            params.padding_value = value;
        }
        // Qwen processors set `padding=true, truncation=false`: n_samples is
        // the default padding target, not an input limit. Honor it only when a
        // checkpoint or deployment explicitly enables truncation. A custom
        // max_samples remains available as an operational hard limit.
        params.max_samples = preprocessor_config
            .get_extra::<usize>("max_samples")
            .or_else(|| {
                preprocessor_config
                    .get_extra::<bool>("truncation")
                    .filter(|enabled| *enabled)
                    .and_then(|_| {
                        preprocessor_config
                            .get_extra::<usize>("n_samples")
                            .or_else(|| {
                                preprocessor_config
                                    .get_extra::<usize>("chunk_length")
                                    .and_then(|seconds| seconds.checked_mul(params.sample_rate))
                            })
                    })
            });

        params
    }
}

fn find_model_usize(config: &Value, paths: &[&[&str]]) -> Option<usize> {
    paths.iter().find_map(|path| {
        let mut value = config;
        for key in *path {
            value = value.get(*key)?;
        }
        value.as_u64().and_then(|value| usize::try_from(value).ok())
    })
}

#[derive(Debug, Clone)]
pub struct Qwen3AudioProcessor {
    params: Qwen3AudioParams,
    /// The window, the filter bank and the FFT plan of these parameters, built on the first clip
    /// (after the parameters are validated) and kept for every clip after it.
    frontend: OnceLock<Arc<MelFrontend>>,
}

impl Default for Qwen3AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3AudioProcessor {
    pub fn new() -> Self {
        Self::with_params(Qwen3AudioParams::default())
    }

    pub(crate) fn with_params(params: Qwen3AudioParams) -> Self {
        Self {
            params,
            frontend: OnceLock::new(),
        }
    }

    pub fn from_configs(model_config: &Value, preprocessor_config: &PreProcessorConfig) -> Self {
        Self::with_params(Qwen3AudioParams::from_configs(
            model_config,
            preprocessor_config,
        ))
    }

    pub fn params(&self) -> &Qwen3AudioParams {
        &self.params
    }

    /// The log-mel features of a batch of decoded clips.
    pub fn preprocess_decoded_clips(
        &self,
        clips: &[&DecodedAudio],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.validate_params()?;
        TransformError::if_empty(clips)?;

        let mut waveforms: Vec<Cow<'_, [f32]>> = Vec::with_capacity(clips.len());
        for (clip_index, clip) in clips.iter().enumerate() {
            if clip.sample_rate == 0 {
                return Err(TransformError::ShapeError(
                    "decoded audio sample rate must be positive".to_string(),
                ));
            }
            if clip.samples.is_empty() {
                return Err(TransformError::ShapeError(
                    "decoded audio contains no samples".to_string(),
                ));
            }
            if !all_samples_finite(&clip.samples) {
                return Err(TransformError::ShapeError(
                    "decoded audio contains a non-finite sample".to_string(),
                ));
            }
            let mut samples: Cow<'_, [f32]> = if clip.sample_rate == self.params.sample_rate {
                Cow::Borrowed(clip.samples.as_slice())
            } else {
                Cow::Owned(bandlimited_resample(
                    &clip.samples,
                    clip.sample_rate,
                    self.params.sample_rate,
                )?)
            };
            if let Some(max_samples) = self.params.max_samples {
                if samples.len() > max_samples {
                    samples = match samples {
                        Cow::Borrowed(borrowed) => Cow::Borrowed(&borrowed[..max_samples]),
                        Cow::Owned(mut owned) => {
                            owned.truncate(max_samples);
                            Cow::Owned(owned)
                        }
                    };
                }
            }
            if samples.is_empty() {
                return Err(TransformError::ShapeError(
                    "decoded audio contains no samples after truncation".to_string(),
                ));
            }
            if samples.len() < self.params.hop_length {
                return Err(TransformError::ShapeError(format!(
                    "Qwen3 audio clip at batch index {clip_index} has {} samples after resampling and truncation; at least {} are required",
                    samples.len(),
                    self.params.hop_length
                )));
            }
            waveforms.push(samples);
        }

        let max_samples = waveforms
            .iter()
            .map(|samples| samples.len())
            .max()
            .unwrap_or(0);
        // Whisper's centered STFT yields floor(samples / hop) + 1 frames and
        // then drops the final frame. This matches padding=True with the batch
        // padded to its longest waveform.
        let max_frames = max_samples / self.params.hop_length;
        if max_frames == 0 {
            return Err(TransformError::ShapeError(format!(
                "Qwen3 audio requires at least {} samples after resampling",
                self.params.hop_length
            )));
        }

        let batch_size = waveforms.len();
        u32::try_from(max_frames).map_err(|_| {
            TransformError::ShapeError("Qwen3 audio frame count exceeds u32".to_string())
        })?;
        let clip_values = self.params.n_mels.checked_mul(max_frames).ok_or_else(|| {
            TransformError::ShapeError("Qwen3 audio feature size overflow".to_string())
        })?;
        let feature_values = batch_size.checked_mul(clip_values).ok_or_else(|| {
            TransformError::ShapeError("Qwen3 audio feature size overflow".to_string())
        })?;
        let mut all_features = Vec::new();
        all_features
            .try_reserve_exact(feature_values)
            .map_err(|e| {
                TransformError::ShapeError(format!("failed to allocate Qwen3 audio features: {e}"))
            })?;
        all_features.resize(feature_values, 0.0);
        let frontend = self
            .frontend
            .get_or_init(|| Arc::new(MelFrontend::build(&self.params)));
        let mut scratch = MelScratch::new(frontend, &self.params);
        let mut attention_mask = Vec::with_capacity(batch_size * max_frames);
        let mut feature_lengths = Vec::with_capacity(batch_size);
        let mut token_counts = Vec::with_capacity(batch_size);
        let mut item_sizes = Vec::with_capacity(batch_size);

        for (features, waveform) in all_features.chunks_exact_mut(clip_values).zip(&waveforms) {
            let feature_length = (waveform.len() / self.params.hop_length).min(max_frames);
            frontend.log_mel(
                waveform,
                max_samples,
                max_frames,
                &self.params,
                &mut scratch,
                features,
            );

            attention_mask.extend((0..max_frames).map(|frame| i64::from(frame < feature_length)));
            feature_lengths.push(feature_length as i64);
            token_counts.push(qwen3_audio_output_length(
                feature_length,
                self.params.n_window,
            ));
            item_sizes.push((self.params.n_mels as u32, feature_length as u32));
        }

        let encoder_input =
            Array3::from_shape_vec((batch_size, self.params.n_mels, max_frames), all_features)
                .map_err(|error| {
                    TransformError::ShapeError(format!(
                "failed to create Qwen3 audio input [{batch_size}, {}, {max_frames}]: {error}",
                self.params.n_mels
            ))
                })?;

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, token_counts, item_sizes)
                .with_extra(
                    "feature_attention_mask",
                    ModelSpecificValue::int_2d(attention_mask, batch_size, max_frames),
                )
                .with_extra(
                    "audio_feature_lengths",
                    ModelSpecificValue::int_1d(feature_lengths),
                ),
        )
    }

    pub fn preprocess_decoded(&self, decoded: DecodedAudio) -> Result<Array2<f32>, TransformError> {
        let output = self.preprocess_decoded_clips(&[&decoded])?;
        output
            .encoder_input
            .into_f32()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|error| TransformError::ShapeError(error.to_string()))?
            .index_axis_move(ndarray::Axis(0), 0)
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|error| TransformError::ShapeError(error.to_string()))
    }

    fn validate_params(&self) -> Result<(), TransformError> {
        if self.params.sample_rate == 0
            || self.params.n_mels == 0
            || self.params.n_fft == 0
            || self.params.hop_length == 0
            || self.params.n_window == 0
        {
            return Err(TransformError::ShapeError(
                "Qwen3 audio sample rate, mel bins, FFT size, hop length, and window size must be positive"
                    .to_string(),
            ));
        }
        if self.params.n_fft < self.params.hop_length {
            return Err(TransformError::ShapeError(format!(
                "Qwen3 audio n_fft ({}) must be at least hop_length ({})",
                self.params.n_fft, self.params.hop_length
            )));
        }
        if !self.params.padding_value.is_finite() {
            return Err(TransformError::ShapeError(
                "Qwen3 audio padding_value must be finite".to_string(),
            ));
        }
        if self.params.max_samples == Some(0) {
            return Err(TransformError::ShapeError(
                "Qwen3 audio max_samples must be positive".to_string(),
            ));
        }
        if self.params.n_fft > MAX_N_FFT
            || self.params.n_mels > MAX_N_MELS
            || self.params.n_window.checked_mul(2).is_none()
        {
            return Err(TransformError::ShapeError(
                "Qwen3 audio parameter exceeds supported bounds".to_string(),
            ));
        }
        Ok(())
    }
}

impl AudioPreProcessor for Qwen3AudioProcessor {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let decoded: Vec<&DecodedAudio> = clips.iter().map(|clip| clip.decoded()).collect();
        self.preprocess_decoded_clips(&decoded)
    }
}

const MAX_N_FFT: usize = 1 << 16;
const MAX_N_MELS: usize = 1 << 12;

#[cfg_attr(
    target_arch = "x86_64",
    multiversion::multiversion(targets("x86_64+avx2"))
)]
fn all_samples_finite(samples: &[f32]) -> bool {
    samples
        .chunks(256)
        .all(|chunk| chunk.iter().fold(true, |ok, s| ok & s.is_finite()))
}

fn qwen_audio_cnn_output_length(mut input_length: usize) -> usize {
    for _ in 0..3 {
        input_length = input_length.div_ceil(2);
    }
    input_length
}

/// Output tokens produced by Qwen's chunked audio encoder for a log-mel length.
fn qwen3_audio_output_length(input_length: usize, n_window: usize) -> usize {
    debug_assert!(n_window > 0);
    let chunk_size = 2 * n_window;
    let full_windows = input_length / chunk_size;
    let remainder = input_length % chunk_size;
    full_windows * qwen_audio_cnn_output_length(chunk_size)
        + qwen_audio_cnn_output_length(remainder)
}

/// Sized so default `power` and `power_cols` fit together inside 32 KiB L1d.
const FRAMES_PER_CHUNK: usize = 16;

#[derive(Clone, Copy)]
#[repr(align(64))]
struct ChunkFrames([f32; FRAMES_PER_CHUNK]);

/// What the log-mel frontend needs beside the samples, built once per configuration: the window,
/// the filter bank with the bin band each filter's weight lies in, and the FFT plan.
struct MelFrontend {
    window: Vec<f32>,
    /// `n_mels` rows of `n_fft / 2 + 1` weights.
    filters: Vec<f32>,
    /// Per filter, the half-open bin range `[first, last)` outside which its weight is zero.
    bands: Vec<(usize, usize)>,
    fft: Arc<dyn Fft<f32>>,
}

impl fmt::Debug for MelFrontend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MelFrontend")
            .field("window", &self.window.len())
            .field("filters", &self.bands.len())
            .field("fft", &self.fft.len())
            .finish()
    }
}

struct MelScratch {
    frames: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    power: Vec<f32>,
    power_cols: Vec<ChunkFrames>,
}

impl MelScratch {
    fn new(frontend: &MelFrontend, params: &Qwen3AudioParams) -> Self {
        let fft_bins = params.n_fft / 2 + 1;
        Self {
            frames: vec![Complex32::new(0.0, 0.0); FRAMES_PER_CHUNK * params.n_fft],
            fft_scratch: vec![Complex32::new(0.0, 0.0); frontend.fft.get_inplace_scratch_len()],
            power: vec![0.0_f32; FRAMES_PER_CHUNK * fft_bins],
            power_cols: vec![ChunkFrames([0.0; FRAMES_PER_CHUNK]); fft_bins],
        }
    }
}

impl MelFrontend {
    fn build(params: &Qwen3AudioParams) -> Self {
        let fft_bins = params.n_fft / 2 + 1;
        let filters = mel_basis(params.sample_rate, params.n_fft, params.n_mels);
        // 394 of the 25,728 weights are non-zero at the default geometry.
        let bands = mel_bands(&filters, params.n_mels, fft_bins);
        Self {
            window: hann_window(params.n_fft),
            filters,
            bands,
            fft: FftPlanner::<f32>::new().plan_fft_forward(params.n_fft),
        }
    }

    /// The normalised log-mel rows of one clip into `out` (`n_mels` rows of `frame_count`
    /// values), as the frame-by-frame computation produced them: the same window, the same FFT
    /// of every frame, the same power term, the same product in the same bin order, so the
    /// bits do not move. Centre reflection sees `samples` extended to `padded_len` with
    /// `padding_value`, matching a padded batch in the reference.
    fn log_mel(
        &self,
        samples: &[f32],
        padded_len: usize,
        frame_count: usize,
        params: &Qwen3AudioParams,
        scratch: &mut MelScratch,
        out: &mut [f32],
    ) {
        debug_assert!(samples.len() <= padded_len);
        debug_assert_eq!(out.len(), self.bands.len() * frame_count);
        let n_fft = params.n_fft;

        let mut peak = f32::NEG_INFINITY;
        let mut frame = 0;
        while frame < frame_count {
            let count = FRAMES_PER_CHUNK.min(frame_count - frame);
            window_chunk(
                samples,
                padded_len,
                params.padding_value,
                params.hop_length,
                &self.window,
                frame,
                &mut scratch.frames[..count * n_fft],
            );
            self.fft.process_with_scratch(
                &mut scratch.frames[..count * n_fft],
                &mut scratch.fft_scratch,
            );

            let chunk_peak = power_and_mel_rows(
                &scratch.frames[..count * n_fft],
                (&mut scratch.power, &mut scratch.power_cols),
                &self.filters,
                &self.bands,
                n_fft,
                (frame, frame_count),
                out,
            );
            peak = peak.max(chunk_peak);
            frame += count;
        }

        clamp_and_scale(out, peak - 8.0);
    }
}

#[cfg_attr(
    target_arch = "x86_64",
    multiversion::multiversion(targets("x86_64+avx2+fma"))
)]
fn window_chunk(
    samples: &[f32],
    padded_len: usize,
    padding_value: f32,
    hop_length: usize,
    window: &[f32],
    frame_start: usize,
    frames_out: &mut [Complex32],
) {
    let n_fft = window.len();
    if n_fft == 0 {
        return;
    }
    let pad = n_fft / 2;
    for (k, dst) in frames_out.chunks_exact_mut(n_fft).enumerate() {
        let start = (frame_start + k) * hop_length;
        if let Some(src) = start
            .checked_sub(pad)
            .and_then(|offset| samples.get(offset..offset.checked_add(n_fft)?))
        {
            for (slot, (&sample, &weight)) in dst.iter_mut().zip(src.iter().zip(window)) {
                *slot = Complex32::new(sample * weight, 0.0);
            }
        } else {
            for (idx, (slot, &weight)) in dst.iter_mut().zip(window).enumerate() {
                let sample = reflected_sample(samples, padded_len, padding_value, pad, start + idx);
                *slot = Complex32::new(sample * weight, 0.0);
            }
        }
    }
}

#[cfg_attr(
    target_arch = "x86_64",
    multiversion::multiversion(targets("x86_64+avx2+fma"))
)]
fn power_and_mel_rows(
    frames: &[Complex32],
    power: (&mut [f32], &mut [ChunkFrames]),
    filters: &[f32],
    bands: &[(usize, usize)],
    n_fft: usize,
    span: (usize, usize),
    out: &mut [f32],
) -> f32 {
    if n_fft == 0 {
        return f32::NEG_INFINITY;
    }
    let (power_rows, power_cols) = power;
    let (frame, frame_count) = span;
    let count = (frames.len() / n_fft).min(FRAMES_PER_CHUNK);
    let fft_bins = n_fft / 2 + 1;
    debug_assert!(
        count > 0 && power_cols.len() >= fft_bins && power_rows.len() >= count * fft_bins
    );
    let (Some(power_cols), Some(power_rows)) = (
        power_cols.get_mut(..fft_bins),
        power_rows.get_mut(..count * fft_bins),
    ) else {
        return f32::NEG_INFINITY;
    };

    // The power term of each bin once per frame; `mul_add`, as the product reads it,
    // not `re * re + im * im`, which rounds twice and differs on real spectra (the
    // sibling frontends hoist the unfused form; this one must not copy it).
    let mut all_finite = true;
    for (row, frame_bins) in power_rows
        .chunks_exact_mut(fft_bins)
        .zip(frames.chunks_exact(n_fft).take(count))
    {
        for (value, bin) in row.iter_mut().zip(frame_bins) {
            let p = bin.re.mul_add(bin.re, bin.im * bin.im);
            *value = p;
            all_finite &= p.is_finite();
        }
    }

    for (bin, col) in power_cols.iter_mut().enumerate() {
        for (slot, row) in col.0[..count]
            .iter_mut()
            .zip(power_rows.chunks_exact(fft_bins))
        {
            *slot = row[bin];
        }
        if count < FRAMES_PER_CHUNK {
            col.0[count..].fill(0.0);
        }
    }

    let mut peaks = [f32::NEG_INFINITY; FRAMES_PER_CHUNK];
    for (mel, &band) in bands.iter().enumerate() {
        // Full-row fallback keeps `0.0 * inf = NaN` without changing finite lanes.
        let (first, last) = if all_finite { band } else { (0, fft_bins) };
        let (Some(weights), Some(cols), Some(dst)) = (
            filters.get(mel * fft_bins + first..mel * fft_bins + last),
            power_cols.get(first..last),
            out.get_mut(mel * frame_count + frame..mel * frame_count + frame + count),
        ) else {
            continue;
        };
        let mut acc = [0.0_f32; FRAMES_PER_CHUNK];
        for (&weight, col) in weights.iter().zip(cols) {
            for (sum, &p) in acc.iter_mut().zip(&col.0) {
                *sum = weight.mul_add(p, *sum);
            }
        }
        for (slot, &value) in dst.iter_mut().zip(&acc) {
            *slot = value.max(1e-10).log10();
        }
        for (peak, &logged) in peaks[..count].iter_mut().zip(&*dst) {
            *peak = peak.max(logged);
        }
    }
    peaks[..count]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
}

#[cfg_attr(
    target_arch = "x86_64",
    multiversion::multiversion(targets("x86_64+avx2"))
)]
fn clamp_and_scale(out: &mut [f32], floor: f32) {
    for value in out {
        *value = (value.max(floor) + 4.0) * 0.25;
    }
}

/// Element `offset` of `samples` extended to `len`, then `numpy` `"reflect"`-padded by `padding`.
#[inline]
fn reflected_sample(
    samples: &[f32],
    len: usize,
    padding_value: f32,
    padding: usize,
    offset: usize,
) -> f32 {
    let value_at = |index: usize| samples.get(index).copied().unwrap_or(padding_value);
    let Some(last) = len.checked_sub(1).filter(|_| padding > 0) else {
        return value_at(offset);
    };
    let Some(period) = last.checked_mul(2).and_then(std::num::NonZeroUsize::new) else {
        return value_at(0);
    };
    let r = offset.abs_diff(padding) % period;
    value_at(if r > last { period.get() - r } else { r })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(samples: usize) -> DecodedAudio {
        DecodedAudio {
            samples: vec![0.0; samples],
            sample_rate: 16_000,
        }
    }

    #[test]
    fn silence_matches_whisper_normalization() {
        let features = Qwen3AudioProcessor::new()
            .preprocess_decoded(decoded(1600))
            .unwrap();
        assert_eq!(features.shape(), &[128, 10]);
        assert!(features.iter().all(|value| (*value + 1.5).abs() < 1e-6));
    }

    #[test]
    fn log_mel_matches_numpy_whisper_reference() {
        let samples = (0..1000)
            .map(|index| ((index % 23) as f32 - 11.0) / 32.0)
            .collect();
        let features = Qwen3AudioProcessor::new()
            .preprocess_decoded(DecodedAudio {
                samples,
                sample_rate: 16_000,
            })
            .unwrap();
        assert_eq!(features.shape(), &[128, 6]);

        for ((mel, frame), expected) in [
            ((0, 0), 0.71467185),
            ((10, 0), 0.735_666_9),
            ((20, 1), 0.36943567),
            ((32, 3), 0.79836977),
            ((64, 5), -0.582_248_9),
            ((100, 5), 0.60166824),
            ((127, 5), 0.52557164),
        ] {
            assert!(
                (features[[mel, frame]] - expected).abs() < 2e-4,
                "log-mel mismatch at ({mel}, {frame}): {} vs {expected}",
                features[[mel, frame]]
            );
        }
        let sum: f64 = features.iter().map(|&value| f64::from(value)).sum();
        assert!((sum - 125.08224487).abs() < 0.02, "feature sum {sum}");
    }

    #[test]
    fn log_mel_boundary_matches_hf_whisper_reference() {
        let samples = (0..1600)
            .map(|index| ((index % 23) as f32 - 11.0) / 32.0)
            .collect();
        let features = Qwen3AudioProcessor::new()
            .preprocess_decoded(DecodedAudio {
                samples,
                sample_rate: 16_000,
            })
            .unwrap();

        assert_eq!(features.shape(), &[128, 10]);
        for (mel, expected) in [(64, 0.017_234_564), (100, 0.601_944_7), (127, 0.525_99)] {
            assert!(
                (features[[mel, 9]] - expected).abs() < 2e-4,
                "last-frame log-mel mismatch at ({mel}, 9): {} vs {expected}",
                features[[mel, 9]]
            );
        }
    }

    #[test]
    fn batches_variable_lengths_with_feature_mask() {
        let output = Qwen3AudioProcessor::new()
            .preprocess_decoded_clips(&[&decoded(1000), &decoded(801)])
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[2, 128, 6]);
        assert_eq!(output.feature_token_counts, vec![1, 1]);
        assert_eq!(output.item_sizes, vec![(128, 6), (128, 5)]);
        assert!(matches!(
            output.model_specific.get("audio_feature_lengths"),
            Some(ModelSpecificValue::IntTensor { data, shape })
                if data == &vec![6, 5] && shape == &vec![2]
        ));
        assert!(matches!(
            output.model_specific.get("feature_attention_mask"),
            Some(ModelSpecificValue::IntTensor { data, shape })
                if data == &vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                    && shape == &vec![2, 6]
        ));
    }

    #[test]
    fn rejects_sub_hop_clip_in_mixed_batch() {
        let error = Qwen3AudioProcessor::new()
            .preprocess_decoded_clips(&[&decoded(320), &decoded(159)])
            .unwrap_err();

        match error {
            TransformError::ShapeError(message) => assert_eq!(
                message,
                "Qwen3 audio clip at batch index 1 has 159 samples after resampling and truncation; at least 160 are required"
            ),
            other => panic!("expected a shape error, got {other}"),
        }
    }

    #[test]
    fn accepts_per_clip_hop_length_boundary() {
        let processor = Qwen3AudioProcessor::new();
        let expected_token_counts = vec![
            qwen3_audio_output_length(
                320 / processor.params().hop_length,
                processor.params().n_window,
            ),
            qwen3_audio_output_length(
                160 / processor.params().hop_length,
                processor.params().n_window,
            ),
        ];
        assert_eq!(expected_token_counts, vec![1, 1]);

        let output = processor
            .preprocess_decoded_clips(&[&decoded(320), &decoded(160)])
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[2, 128, 2]);
        assert_eq!(output.feature_token_counts, expected_token_counts);
        assert_eq!(output.item_sizes, vec![(128, 2), (128, 1)]);
    }

    #[test]
    fn reads_preprocessor_and_nested_model_config() {
        let preprocessor = PreProcessorConfig::from_json(
            r#"{"sampling_rate": 16000, "feature_size": 64, "n_fft": 320, "hop_length": 80, "n_samples": 4800000}"#,
        )
        .unwrap();
        let params = Qwen3AudioParams::from_configs(
            &serde_json::json!({
                "thinker_config": {"audio_config": {"num_mel_bins": 96, "n_window": 64}}
            }),
            &preprocessor,
        );
        assert_eq!(params.sample_rate, 16_000);
        assert_eq!(params.n_mels, 64);
        assert_eq!(params.n_fft, 320);
        assert_eq!(params.hop_length, 80);
        assert_eq!(params.n_window, 64);
        assert_eq!(params.max_samples, None);

        let truncating_preprocessor = PreProcessorConfig::from_json(
            r#"{"sampling_rate": 16000, "n_samples": 4800000, "truncation": true}"#,
        )
        .unwrap();
        let params =
            Qwen3AudioParams::from_configs(&serde_json::json!({}), &truncating_preprocessor);
        assert_eq!(params.max_samples, Some(4_800_000));

        let params = Qwen3AudioParams::from_configs(
            &serde_json::json!({
                "thinker_config": {"audio_config": {"num_mel_bins": 96}}
            }),
            &PreProcessorConfig::default(),
        );
        assert_eq!(params.n_mels, 96);
        assert_eq!(params.max_samples, None);
    }

    #[test]
    fn n_samples_is_padding_target_not_implicit_truncation() {
        let preprocessor = PreProcessorConfig::from_json(r#"{"n_samples": 320}"#).unwrap();
        let processor = Qwen3AudioProcessor::from_configs(&serde_json::json!({}), &preprocessor);
        let output = processor
            .preprocess_decoded_clips(&[&decoded(800)])
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[1, 128, 5]);
        assert_eq!(output.item_sizes, vec![(128, 5)]);
    }

    #[test]
    fn truncates_to_explicit_audio_limit() {
        let processor = Qwen3AudioProcessor::with_params(Qwen3AudioParams {
            max_samples: Some(320),
            ..Default::default()
        });
        let output = processor
            .preprocess_decoded_clips(&[&decoded(800)])
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[1, 128, 2]);
        assert_eq!(output.feature_token_counts, vec![1]);
        assert_eq!(output.item_sizes, vec![(128, 2)]);
    }

    #[test]
    fn qwen_chunked_encoder_output_lengths_match_reference_formula() {
        assert_eq!(qwen3_audio_output_length(0, 50), 0);
        assert_eq!(qwen3_audio_output_length(1, 50), 1);
        assert_eq!(qwen3_audio_output_length(8, 50), 1);
        assert_eq!(qwen3_audio_output_length(9, 50), 2);
        assert_eq!(qwen3_audio_output_length(99, 50), 13);
        assert_eq!(qwen3_audio_output_length(100, 50), 13);
        assert_eq!(qwen3_audio_output_length(101, 50), 14);
        assert_eq!(qwen3_audio_output_length(3000, 50), 390);
        assert_eq!(qwen3_audio_output_length(17, 4), 3);
    }

    #[test]
    fn reflect_padding_matches_numpy_convention() {
        let pad = |s: &[f32], len, p| -> Vec<f32> {
            (0..len + 2 * p)
                .map(|i| reflected_sample(s, len, 0.0, p, i))
                .collect()
        };
        assert_eq!(
            pad(&[1.0, 2.0, 3.0], 3, 2),
            [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
        );
        assert_eq!(pad(&[2.0], 1, 2), [2.0; 5]);
        // A clip shorter than the batch's longest is extended with the padding value first, so
        // the reflection at its end mirrors the padding.
        assert_eq!(
            pad(&[1.0, 2.0], 4, 2),
            [0.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0]
        );
        assert_eq!(pad(&[1.0, 2.0], 3, 0), [1.0, 2.0, 0.0]);
    }

    /// The frontend as it computed before the frames were batched: the padded waveform copied,
    /// one FFT call per frame with its own scratch, the power term inside the product, and the
    /// product over every bin. The batched frontend must equal it bit for bit on every input.
    fn frame_by_frame_reference(
        samples: &[f32],
        padded_len: usize,
        frame_count: usize,
        params: &Qwen3AudioParams,
    ) -> Vec<f32> {
        let mut waveform = samples.to_vec();
        waveform.resize(padded_len, params.padding_value);
        let pad = params.n_fft / 2;
        let padded: Vec<f32> = (-(pad as isize)..(padded_len + pad) as isize)
            .map(|mut i| {
                let last = padded_len as isize - 1;
                while last > 0 && (i < 0 || i > last) {
                    i = if i < 0 { -i } else { 2 * last - i };
                }
                waveform[if last == 0 { 0 } else { i as usize }]
            })
            .collect();
        let fft_bins = params.n_fft / 2 + 1;
        let window = hann_window(params.n_fft);
        let mel_filters = mel_basis(params.sample_rate, params.n_fft, params.n_mels);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(params.n_fft);
        let mut buffer = vec![Complex32::new(0.0, 0.0); params.n_fft];
        let mut output = vec![0.0_f32; params.n_mels * frame_count];
        for frame in 0..frame_count {
            let start = frame * params.hop_length;
            for index in 0..params.n_fft {
                buffer[index] = Complex32::new(padded[start + index] * window[index], 0.0);
            }
            fft.process(&mut buffer);
            for mel in 0..params.n_mels {
                let filter = &mel_filters[mel * fft_bins..(mel + 1) * fft_bins];
                let mut value = 0.0_f32;
                for bin in 0..fft_bins {
                    let fft_value = buffer[bin];
                    let power = fft_value
                        .re
                        .mul_add(fft_value.re, fft_value.im * fft_value.im);
                    value = filter[bin].mul_add(power, value);
                }
                output[mel * frame_count + frame] = value.max(1e-10).log10();
            }
        }
        let peak = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let floor = peak - 8.0;
        for value in &mut output {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
        output
    }

    fn waveform_corpus() -> Vec<(String, Vec<f32>)> {
        let mut state = 0x243f_6a88_u32;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut corpus: Vec<(String, Vec<f32>)> = vec![
            (
                "tone".into(),
                (0..16_000)
                    .map(|i| (i as f32 * 0.001).sin() * 0.5)
                    .collect(),
            ),
            ("noise".into(), (0..16_000).map(|_| noise()).collect()),
            ("silence".into(), vec![0.0; 8_000]),
            ("impulse".into(), {
                let mut v = vec![0.0_f32; 8_000];
                v[7] = 1.0;
                v
            }),
            (
                "loud".into(),
                (0..8_000)
                    .map(|i| {
                        if (i / 37) % 2 == 0 {
                            30_000.0
                        } else {
                            -30_000.0
                        }
                    })
                    .collect(),
            ),
            (
                "quiet".into(),
                (0..8_000)
                    .map(|i| (i as f32 * 0.01).sin() * 1e-30)
                    .collect(),
            ),
        ];
        for frames in [1, 15, 16, 17, 31, 32, 33, 65] {
            let count = frames * 160 + 7;
            corpus.push((
                format!("{frames}_frames"),
                (0..count)
                    .map(|i| (i as f32 * 0.003).sin() * 0.3 + noise() * 0.05)
                    .collect(),
            ));
        }
        corpus
    }

    fn log_mel_of(processor: &Qwen3AudioProcessor, samples: &[f32]) -> Vec<f32> {
        processor
            .preprocess_decoded(DecodedAudio {
                samples: samples.to_vec(),
                sample_rate: 16_000,
            })
            .expect("log-mel")
            .into_raw_vec_and_offset()
            .0
    }

    fn assert_same_bits(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{name}: value count");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.to_bits() == e.to_bits(),
                "{name}: value {index} is {a:?}, the frame-by-frame computation gives {e:?}"
            );
        }
    }

    #[test]
    fn the_batched_frontend_equals_the_frame_by_frame_computation_bit_for_bit() {
        let processor = Qwen3AudioProcessor::new();
        let params = processor.params().clone();
        for (name, samples) in waveform_corpus() {
            let frames = samples.len() / params.hop_length;
            assert_same_bits(
                &name,
                &log_mel_of(&processor, &samples),
                &frame_by_frame_reference(&samples, samples.len(), frames, &params),
            );
        }
    }

    #[test]
    fn a_batch_of_unequal_clips_equals_each_clip_padded_to_the_longest() {
        let processor = Qwen3AudioProcessor::new();
        let params = processor.params().clone();
        let corpus = waveform_corpus();
        let clips: Vec<DecodedAudio> = corpus
            .iter()
            .map(|(_, samples)| DecodedAudio {
                samples: samples.clone(),
                sample_rate: 16_000,
            })
            .collect();
        let refs: Vec<&DecodedAudio> = clips.iter().collect();
        let output = processor.preprocess_decoded_clips(&refs).expect("batch");
        let longest = corpus.iter().map(|(_, s)| s.len()).max().unwrap_or(0);
        let frames = longest / params.hop_length;
        assert_eq!(output.encoder_input.shape(), &[corpus.len(), 128, frames]);
        let values = output.encoder_input.as_f32();
        let values = values.as_slice_memory_order().expect("contiguous");
        for (index, (name, samples)) in corpus.iter().enumerate() {
            assert_same_bits(
                name,
                &values[index * 128 * frames..(index + 1) * 128 * frames],
                &frame_by_frame_reference(samples, longest, frames, &params),
            );
        }
    }

    #[test]
    fn a_spectrum_that_overflows_takes_the_full_product_and_keeps_its_bits() {
        // f32 squares overflow above about 1.8e19; the windowed FFT of 1e30-scale samples is
        // far past it, so every power term is infinite and the band product would differ from
        // the full one (a zero weight times infinity is NaN there).
        let processor = Qwen3AudioProcessor::new();
        let params = processor.params().clone();
        let overflowing: Vec<f32> = (0..4_000)
            .map(|i| if i % 2 == 0 { 1e30 } else { -1e30 })
            .collect();
        assert!(
            a_power_term_overflows(&overflowing, &params),
            "the overflow fixture no longer overflows; the full-product path is untested"
        );
        let frames = overflowing.len() / params.hop_length;
        let expected = frame_by_frame_reference(&overflowing, overflowing.len(), frames, &params);
        // The full product turns every row into NaN (a zero weight times infinity), which
        // `max` then drops for the floor, so the clip reads as silence; the band product alone
        // would read as infinity. Both frontends must agree on the former.
        assert!(
            expected.iter().all(|value| *value == -1.5),
            "the reference is the floor"
        );
        assert_same_bits("overflow", &log_mel_of(&processor, &overflowing), &expected);

        let mut mixed: Vec<f32> = (0..4_000).map(|i| (i as f32 * 0.03).sin() * 0.25).collect();
        mixed[800..960].copy_from_slice(&overflowing[..160]);
        let mixed_expected = frame_by_frame_reference(&mixed, mixed.len(), frames, &params);
        assert_same_bits(
            "mixed_chunk",
            &log_mel_of(&processor, &mixed),
            &mixed_expected,
        );

        // And an ordinary clip takes the band product throughout.
        let (_, ordinary) = waveform_corpus().swap_remove(1);
        assert!(
            !a_power_term_overflows(&ordinary, &params),
            "an ordinary clip should take the band product"
        );
    }

    #[test]
    fn non_default_geometries_and_reflection_edges_keep_their_bits() {
        for (n_fft, hop_length, n_mels, padding_value, len) in [
            (320, 80, 64, 0.0, 2_573),
            (257, 64, 48, -0.125, 1_900),
            (400, 160, 128, 0.25, 160),
            (400, 160, 128, -0.5, 300),
            (8, 1, 4, 0.0, 1),
            (1, 1, 4, 0.0, 19),
        ] {
            let params = Qwen3AudioParams {
                n_mels,
                n_fft,
                hop_length,
                padding_value,
                ..Qwen3AudioParams::default()
            };
            let samples: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).cos() * 0.4).collect();
            let actual = log_mel_of(&Qwen3AudioProcessor::with_params(params.clone()), &samples);
            let expected = frame_by_frame_reference(&samples, len, len / hop_length, &params);
            assert_same_bits(&format!("{n_fft}/{hop_length}/{len}"), &actual, &expected);
        }
        for (n_fft, n_mels, n_window) in [
            (MAX_N_FFT + 1, 128, 50),
            (400, MAX_N_MELS + 1, 50),
            (400, 128, usize::MAX),
        ] {
            let bad = Qwen3AudioParams {
                n_mels,
                n_fft,
                n_window,
                ..Qwen3AudioParams::default()
            };
            let err = Qwen3AudioProcessor::with_params(bad).preprocess_decoded(decoded(320));
            assert!(err.is_err());
        }
    }

    /// One band per filter describes the filter only while its weight is contiguous. It is,
    /// because the filters are triangular and adjacent, and the band product would silently
    /// drop weight if that ever stopped being true, so it is asserted rather than trusted;
    /// the bank's density is the reason the band product pays.
    #[test]
    fn mel_filter_bands_are_contiguous_and_sparse() {
        let params = Qwen3AudioParams::default();
        let fft_bins = params.n_fft / 2 + 1;
        let frontend = MelFrontend::build(&params);
        let mut non_zero = 0;
        for (mel, &(first, last)) in frontend.bands.iter().enumerate() {
            let row = &frontend.filters[mel * fft_bins..(mel + 1) * fft_bins];
            assert!(
                row[..first].iter().chain(&row[last..]).all(|w| *w == 0.0),
                "mel row {mel} carries weight outside [{first}, {last}); the band product \
                 would drop it"
            );
            non_zero += row.iter().filter(|w| **w != 0.0).count();
        }
        // 394 of 25,728 at the default geometry.
        assert!(
            non_zero * 50 < params.n_mels * fft_bins,
            "the filter bank is {non_zero}/{} non-zero, denser than the band product assumes",
            params.n_mels * fft_bins
        );
    }

    /// `mul_add` and the unfused product round differently on real spectra, so the comment
    /// warning against the sibling frontends' formulation is load bearing. If this ever stops
    /// differing the warning can go.
    #[test]
    fn the_fused_and_unfused_power_terms_really_do_differ() {
        let params = Qwen3AudioParams::default();
        let (_, samples) = waveform_corpus().swap_remove(1);
        let window = hann_window(params.n_fft);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(params.n_fft);
        let mut buffer = vec![Complex32::new(0.0, 0.0); params.n_fft];
        let (mut differing, mut total) = (0_usize, 0_usize);
        for frame in 0..40 {
            let start = frame * params.hop_length;
            for index in 0..params.n_fft {
                buffer[index] = Complex32::new(samples[start + index] * window[index], 0.0);
            }
            fft.process(&mut buffer);
            for bin in &buffer[..params.n_fft / 2 + 1] {
                let fused = bin.re.mul_add(bin.re, bin.im * bin.im);
                let unfused = bin.re * bin.re + bin.im * bin.im;
                total += 1;
                differing += usize::from(fused.to_bits() != unfused.to_bits());
            }
        }
        assert!(
            differing > 0,
            "the two power formulations agreed on all {total} bins"
        );
    }

    /// Whether some frame's power spectrum of `samples` is not finite, so the frontend takes the
    /// full product on it.
    fn a_power_term_overflows(samples: &[f32], params: &Qwen3AudioParams) -> bool {
        let window = hann_window(params.n_fft);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(params.n_fft);
        let mut buffer = vec![Complex32::new(0.0, 0.0); params.n_fft];
        (0..samples.len() / params.hop_length).any(|frame| {
            let start = frame * params.hop_length;
            for index in 0..params.n_fft {
                let sample = samples.get(start + index).copied().unwrap_or(0.0);
                buffer[index] = Complex32::new(sample * window[index], 0.0);
            }
            fft.process(&mut buffer);
            buffer[..params.n_fft / 2 + 1]
                .iter()
                .any(|bin| !bin.re.mul_add(bin.re, bin.im * bin.im).is_finite())
        })
    }

    #[test]
    fn a_processor_clone_shares_the_built_frontend_and_computes_the_same_bits() {
        let processor = Qwen3AudioProcessor::new();
        let (_, samples) = waveform_corpus().swap_remove(1);
        let before = log_mel_of(&processor, &samples);
        let clone = processor.clone();
        assert_same_bits("clone", &log_mel_of(&clone, &samples), &before);
        assert!(
            Arc::ptr_eq(
                processor.frontend.get().expect("built"),
                clone.frontend.get().expect("built")
            ),
            "a clone after the first clip carries the same tables"
        );
    }
}
