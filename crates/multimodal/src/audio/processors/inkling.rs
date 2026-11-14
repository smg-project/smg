//! Inkling audio preprocessing.
//!
//! Implements the Inkling feature-extraction pipeline for the model-facing part:
//! audio bytes are decoded to mono f32, resampled to the configured sample
//! rate, non-zero quiet signals are raised to the configured RMS floor,
//! transformed to Slaney-normalized log-mel magnitudes, then quantized to dMel
//! bin ids. The returned encoder input stores those integer bin ids as f32,
//! matching the checkpoint feature-extractor contract and allowing the transport
//! to apply the same configurable floating-point wire dtype as other modalities.

use std::{
    borrow::Cow,
    fmt,
    sync::{Arc, OnceLock},
};

use ndarray::Array2;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::{
    audio::{
        transforms::{bandlimited_resample, hann_window, mel_bands, mel_basis},
        AudioPreProcessor, DecodedAudio,
    },
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
};

#[derive(Debug, Clone)]
pub struct InklingAudioEncoderParams {
    pub sample_rate: usize,
    pub window_size_multiplier: f64,
    pub n_fft: Option<usize>,
    pub n_mels: usize,
    pub num_dmel_bins: usize,
    pub dmel_min_value: f64,
    pub dmel_max_value: f64,
    pub audio_token_duration_s: f64,
    pub audio_rms_norm_floor: f64,
}

impl Default for InklingAudioEncoderParams {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            window_size_multiplier: 2.0,
            n_fft: None,
            n_mels: 80,
            num_dmel_bins: 16,
            dmel_min_value: -7.0,
            dmel_max_value: 2.0,
            audio_token_duration_s: 0.05,
            audio_rms_norm_floor: 0.01,
        }
    }
}

impl InklingAudioEncoderParams {
    pub fn from_model_config(config: &serde_json::Value) -> Self {
        let mut params = Self::default();
        let Some(audio_config) = config.get("audio_config") else {
            return params;
        };
        if let Some(v) = audio_config.get("n_mel_bins").and_then(|v| v.as_u64()) {
            params.n_mels = v as usize;
        }
        if let Some(v) = audio_config.get("mel_vocab_size").and_then(|v| v.as_u64()) {
            params.num_dmel_bins = v as usize;
        }
        if let Some(v) = audio_config.get("dmel_min_value").and_then(|v| v.as_f64()) {
            params.dmel_min_value = v;
        }
        if let Some(v) = audio_config.get("dmel_max_value").and_then(|v| v.as_f64()) {
            params.dmel_max_value = v;
        }
        if let Some(v) = audio_config
            .get("audio_rms_norm_floor")
            .and_then(|v| v.as_f64())
        {
            params.audio_rms_norm_floor = v;
        }
        params
    }

    fn hop_length(&self) -> Result<usize, TransformError> {
        exact_sample_count(
            self.audio_token_duration_s * self.sample_rate as f64,
            "audio_token_duration_s * sample_rate",
        )
    }

    fn window_size(&self) -> Result<usize, TransformError> {
        exact_sample_count(
            self.audio_token_duration_s * self.window_size_multiplier * self.sample_rate as f64,
            "audio_token_duration_s * window_size_multiplier * sample_rate",
        )
    }
}

/// The frames of a clip run through the FFT so many at a time: the chunk's
/// complex frames stay in cache and one FFT call covers them all.
const FRAMES_PER_CHUNK: usize = 32;

#[derive(Debug, Clone)]
pub struct InklingAudioProcessor {
    params: InklingAudioEncoderParams,
    /// The tables and the FFT plan of `params`, built on the first clip and
    /// shared by every clone of the processor.
    frontend: OnceLock<Arc<DmelFrontend>>,
}

impl Default for InklingAudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl InklingAudioProcessor {
    pub fn new() -> Self {
        Self::with_params(InklingAudioEncoderParams::default())
    }

    fn with_params(params: InklingAudioEncoderParams) -> Self {
        Self {
            params,
            frontend: OnceLock::new(),
        }
    }

    pub fn from_model_config(config: &serde_json::Value) -> Self {
        Self::with_params(InklingAudioEncoderParams::from_model_config(config))
    }

    /// The dMel bin ids of a batch of decoded clips, `[tokens, n_mels]` over
    /// the clips in order. The clips are borrowed: each is copied once, into
    /// the batch's padded buffer, where the RMS floor is applied; a clip at
    /// another rate is resampled into a buffer of its own first.
    pub fn preprocess_decoded_clips(
        &self,
        clips: &[&DecodedAudio],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if clips.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let frontend = self.frontend()?;
        let mut waveforms: Vec<Cow<'_, [f32]>> = Vec::with_capacity(clips.len());
        for clip in clips {
            waveforms.push(self.at_frontend_rate(clip)?);
        }
        let n_mels = self.params.n_mels;
        let frame_counts: Vec<usize> = waveforms
            .iter()
            .map(|waveform| frontend.frame_count(waveform.len()))
            .collect();
        let total_tokens = frame_counts
            .iter()
            .try_fold(0_usize, |sum, frames| sum.checked_add(*frames))
            .ok_or_else(|| {
                TransformError::ShapeError("Inkling dMel token count overflow".to_string())
            })?;
        let values = total_tokens.checked_mul(n_mels).ok_or_else(|| {
            TransformError::ShapeError("Inkling dMel bin count overflow".to_string())
        })?;
        let max_samples = waveforms
            .iter()
            .map(|waveform| waveform.len())
            .max()
            .unwrap_or(0);
        let mut scratch = DmelScratch::new(frontend, max_samples);
        let mut all_bins = vec![0.0_f32; values];
        let mut token_counts = Vec::with_capacity(clips.len());
        let mut item_sizes = Vec::with_capacity(clips.len());
        let mut tokens_per_item = Vec::with_capacity(clips.len());
        let mut written = 0;
        for (waveform, &frames) in waveforms.iter().zip(&frame_counts) {
            let out = &mut all_bins[written..written + frames * n_mels];
            frontend.dmel_bins_into(waveform, &self.params, &mut scratch, out)?;
            written += frames * n_mels;
            token_counts.push(frames);
            tokens_per_item.push(frames as i64);
            item_sizes.push((n_mels as u32, frames as u32));
        }

        let encoder_input =
            Array2::from_shape_vec((total_tokens, n_mels), all_bins).map_err(|e| {
                TransformError::ShapeError(format!(
                    "failed to create Inkling audio encoder input [{total_tokens}, {n_mels}]: {e}"
                ))
            })?;

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, token_counts, item_sizes).with_extra(
                "tokens_per_item",
                ModelSpecificValue::int_1d(tokens_per_item),
            ),
        )
    }

    /// The dMel bin ids of one clip, `[frames, n_mels]`.
    pub fn preprocess_decoded(
        &self,
        decoded: &DecodedAudio,
    ) -> Result<Array2<f32>, TransformError> {
        let frontend = self.frontend()?;
        let waveform = self.at_frontend_rate(decoded)?;
        let frames = frontend.frame_count(waveform.len());
        let n_mels = self.params.n_mels;
        let mut scratch = DmelScratch::new(frontend, waveform.len());
        let mut bins = vec![0.0_f32; frames * n_mels];
        frontend.dmel_bins_into(&waveform, &self.params, &mut scratch, &mut bins)?;
        Array2::from_shape_vec((frames, n_mels), bins).map_err(|e| {
            TransformError::ShapeError(format!(
                "failed to create Inkling dMel bins [{frames}, {n_mels}]: {e}"
            ))
        })
    }

    /// The clip's samples at the frontend's rate: borrowed when the clip is
    /// at it already, resampled otherwise.
    fn at_frontend_rate<'a>(
        &self,
        decoded: &'a DecodedAudio,
    ) -> Result<Cow<'a, [f32]>, TransformError> {
        if decoded.sample_rate == 0 {
            return Err(TransformError::ShapeError(
                "decoded audio sample rate must be positive".to_string(),
            ));
        }
        if decoded.samples.is_empty() {
            return Err(TransformError::ShapeError(
                "decoded audio contains no samples".to_string(),
            ));
        }
        if decoded.sample_rate == self.params.sample_rate {
            Ok(Cow::Borrowed(decoded.samples.as_slice()))
        } else {
            Ok(Cow::Owned(bandlimited_resample(
                &decoded.samples,
                decoded.sample_rate,
                self.params.sample_rate,
            )?))
        }
    }

    /// The frontend of this processor's parameters, built once; the
    /// parameters are checked on every call until a build succeeds.
    fn frontend(&self) -> Result<&Arc<DmelFrontend>, TransformError> {
        if let Some(frontend) = self.frontend.get() {
            return Ok(frontend);
        }
        let built = Arc::new(DmelFrontend::build(&self.params)?);
        Ok(self.frontend.get_or_init(|| built))
    }
}

impl AudioPreProcessor for InklingAudioProcessor {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let decoded: Vec<&DecodedAudio> = clips.iter().map(|clip| clip.decoded()).collect();
        self.preprocess_decoded_clips(&decoded)
    }
}

fn exact_sample_count(value: f64, name: &str) -> Result<usize, TransformError> {
    let rounded = value.round();
    if (value - rounded).abs() > 1e-6 {
        return Err(TransformError::ShapeError(format!(
            "{name} must resolve to an integer sample count, got {value}"
        )));
    }
    if rounded <= 0.0 {
        return Err(TransformError::ShapeError(format!(
            "{name} must be positive, got {rounded}"
        )));
    }
    Ok(rounded as usize)
}

fn normalize_audio_rms(samples: &mut [f32], floor: f64) -> Result<(), TransformError> {
    if !floor.is_finite() || floor < 0.0 {
        return Err(TransformError::ShapeError(format!(
            "audio_rms_norm_floor must be finite and non-negative, got {floor}"
        )));
    }
    if floor == 0.0 {
        return Ok(());
    }

    let mean_square = samples
        .iter()
        .map(|&sample| {
            let sample = f64::from(sample);
            sample * sample
        })
        .sum::<f64>()
        / samples.len() as f64;
    let rms = mean_square.sqrt();
    if rms > 0.0 && rms < floor {
        let scale = floor / rms;
        for sample in samples {
            *sample = (f64::from(*sample) * scale) as f32;
        }
    }
    Ok(())
}

/// What the dMel frontend needs beside the samples, built once per
/// configuration: the frame geometry, the window, the filter bank with the
/// bin band each filter's weight lies in, and the FFT plan.
struct DmelFrontend {
    hop_length: usize,
    window_size: usize,
    n_fft: usize,
    fft_bins: usize,
    window: Vec<f32>,
    /// `n_mels` rows of `fft_bins` weights.
    filters: Vec<f32>,
    /// Per filter, the half-open bin range outside which its weight is zero.
    bands: Vec<(usize, usize)>,
    fft: Arc<dyn Fft<f32>>,
}

impl fmt::Debug for DmelFrontend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DmelFrontend")
            .field("hop_length", &self.hop_length)
            .field("window_size", &self.window_size)
            .field("n_fft", &self.n_fft)
            .field("filters", &self.bands.len())
            .finish()
    }
}

/// The buffers one batch reuses for every clip: the padded waveform, a chunk
/// of complex frames, the FFT's own scratch and the chunk's magnitudes.
struct DmelScratch {
    padded: Vec<f32>,
    frames: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    magnitudes: Vec<f32>,
}

impl DmelScratch {
    fn new(frontend: &DmelFrontend, max_samples: usize) -> Self {
        Self {
            padded: Vec::with_capacity(frontend.padding(max_samples).1),
            frames: vec![Complex32::new(0.0, 0.0); FRAMES_PER_CHUNK * frontend.n_fft],
            fft_scratch: vec![Complex32::new(0.0, 0.0); frontend.fft.get_inplace_scratch_len()],
            magnitudes: vec![0.0_f32; FRAMES_PER_CHUNK * frontend.fft_bins],
        }
    }
}

impl DmelFrontend {
    fn build(params: &InklingAudioEncoderParams) -> Result<Self, TransformError> {
        if params.n_mels == 0 || params.num_dmel_bins == 0 {
            return Err(TransformError::ShapeError(
                "n_mels and num_dmel_bins must be positive".to_string(),
            ));
        }
        let hop_length = params.hop_length()?;
        let window_size = params.window_size()?;
        let n_fft = params.n_fft.unwrap_or(window_size);
        if n_fft < window_size {
            return Err(TransformError::ShapeError(format!(
                "n_fft ({n_fft}) must be greater than or equal to window_size ({window_size})"
            )));
        }
        if hop_length == 0 || window_size == 0 {
            return Err(TransformError::ShapeError(
                "hop length and window size must be positive".to_string(),
            ));
        }
        let fft_bins = n_fft / 2 + 1;
        let filters = mel_basis(params.sample_rate, n_fft, params.n_mels);
        let bands = mel_bands(&filters, params.n_mels, fft_bins);
        Ok(Self {
            hop_length,
            window_size,
            n_fft,
            fft_bins,
            window: hann_window(window_size),
            filters,
            bands,
            fft: FftPlanner::<f32>::new().plan_fft_forward(n_fft),
        })
    }

    /// The left padding of a clip of `samples` and its padded length: a
    /// frame's worth less a hop before the clip, and the clip rounded up to
    /// whole hops after it.
    fn padding(&self, samples: usize) -> (usize, usize) {
        let right_pad = samples.div_ceil(self.hop_length) * self.hop_length - samples;
        let left_pad = self.n_fft.saturating_sub(self.hop_length);
        (left_pad, left_pad + samples + right_pad)
    }

    /// The frames (one per token) a clip of `samples` yields.
    fn frame_count(&self, samples: usize) -> usize {
        let (_, padded_len) = self.padding(samples);
        (padded_len - self.n_fft) / self.hop_length + 1
    }

    /// The dMel bin ids of one clip into `out` (`frames` rows of `n_mels`),
    /// as the frame-by-frame computation produced them: the same window, the
    /// same FFT of every frame, the same magnitude term, the same product in
    /// the same bin order, the same quantization, so the bits do not move.
    fn dmel_bins_into(
        &self,
        samples: &[f32],
        params: &InklingAudioEncoderParams,
        scratch: &mut DmelScratch,
        out: &mut [f32],
    ) -> Result<(), TransformError> {
        let (left_pad, padded_len) = self.padding(samples.len());
        scratch.padded.clear();
        scratch.padded.resize(padded_len, 0.0);
        let region = &mut scratch.padded[left_pad..left_pad + samples.len()];
        region.copy_from_slice(samples);
        normalize_audio_rms(region, params.audio_rms_norm_floor)?;

        let n_fft = self.n_fft;
        let fft_bins = self.fft_bins;
        let n_mels = params.n_mels;
        let frame_count = (padded_len - n_fft) / self.hop_length + 1;
        if out.len() != frame_count * n_mels {
            return Err(TransformError::ShapeError(format!(
                "Inkling dMel output holds {} values, {} needed",
                out.len(),
                frame_count * n_mels
            )));
        }
        let mut frame = 0;
        while frame < frame_count {
            let count = FRAMES_PER_CHUNK.min(frame_count - frame);
            for k in 0..count {
                let start = (frame + k) * self.hop_length;
                let (windowed, rest) =
                    scratch.frames[k * n_fft..(k + 1) * n_fft].split_at_mut(self.window_size);
                for (slot, (&sample, &weight)) in windowed.iter_mut().zip(
                    scratch.padded[start..start + self.window_size]
                        .iter()
                        .zip(&self.window),
                ) {
                    *slot = Complex32::new(sample * weight, 0.0);
                }
                rest.fill(Complex32::new(0.0, 0.0));
            }
            self.fft.process_with_scratch(
                &mut scratch.frames[..count * n_fft],
                &mut scratch.fft_scratch,
            );
            for k in 0..count {
                let spectrum = &scratch.frames[k * n_fft..k * n_fft + fft_bins];
                let magnitudes = &mut scratch.magnitudes[k * fft_bins..(k + 1) * fft_bins];
                for (magnitude, bin) in magnitudes.iter_mut().zip(spectrum) {
                    *magnitude = (bin.re.mul_add(bin.re, bin.im * bin.im)).max(1e-10).sqrt();
                }
            }
            // The mel product over each filter's band: a zero weight adds
            // nothing to a finite magnitude, so the band's sum is the full
            // row's sum, bit for bit. Where a magnitude overflowed, a zero
            // weight times infinity is NaN and the full row keeps it.
            for k in 0..count {
                let magnitudes = &scratch.magnitudes[k * fft_bins..(k + 1) * fft_bins];
                let finite = magnitudes.iter().all(|magnitude| magnitude.is_finite());
                let row = &mut out[(frame + k) * n_mels..(frame + k + 1) * n_mels];
                for (mel, slot) in row.iter_mut().enumerate() {
                    let (first, last) = if finite {
                        self.bands[mel]
                    } else {
                        (0, fft_bins)
                    };
                    let weights = &self.filters[mel * fft_bins + first..mel * fft_bins + last];
                    let mut value = 0.0_f32;
                    for (weight, magnitude) in weights.iter().zip(&magnitudes[first..last]) {
                        value += weight * magnitude;
                    }
                    let log_mel = f64::from(value.max(1e-10).log10())
                        .clamp(params.dmel_min_value, params.dmel_max_value);
                    *slot = quantize_dmel(log_mel, params) as f32;
                }
            }
            frame += count;
        }
        Ok(())
    }
}

fn quantize_dmel(value: f64, params: &InklingAudioEncoderParams) -> usize {
    if params.num_dmel_bins <= 1 {
        return 0;
    }
    let span = params.dmel_max_value - params.dmel_min_value;
    if span <= 0.0 {
        return 0;
    }
    let scaled = (value - params.dmel_min_value) / span * (params.num_dmel_bins - 1) as f64;
    // Nearest-bin selection keeps the lower bin on an exact tie.
    (scaled - 0.5)
        .ceil()
        .clamp(0.0, (params.num_dmel_bins - 1) as f64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;

    #[test]
    fn rejects_empty_or_invalid_rate_audio() {
        let processor = InklingAudioProcessor::new();
        assert!(processor
            .preprocess_decoded(&DecodedAudio {
                samples: Vec::new(),
                sample_rate: 16_000,
            })
            .is_err());
        assert!(processor
            .preprocess_decoded(&DecodedAudio {
                samples: vec![0.0],
                sample_rate: 0,
            })
            .is_err());
    }

    #[test]
    fn model_config_overrides_audio_shape_params() {
        let config = serde_json::json!({
            "audio_config": {
                "n_mel_bins": 8,
                "mel_vocab_size": 4,
                "dmel_min_value": -5.0,
                "dmel_max_value": 1.0,
                "audio_rms_norm_floor": 0.02
            }
        });
        let processor = InklingAudioProcessor::from_model_config(&config);
        assert_eq!(processor.params.n_mels, 8);
        assert_eq!(processor.params.num_dmel_bins, 4);
        assert_eq!(processor.params.dmel_min_value, -5.0);
        assert_eq!(processor.params.dmel_max_value, 1.0);
        assert_eq!(processor.params.audio_rms_norm_floor, 0.02);
    }

    fn seeded_signal(scale: f32) -> Vec<f32> {
        (0..1600)
            .map(|i| {
                let raw = (i * 73 + 17 * 977) % 65_536 - 32_768;
                raw as f32 / 32_768.0 * scale
            })
            .collect()
    }

    #[test]
    fn quiet_nonzero_audio_is_normalized_to_rms_floor() {
        let processor = InklingAudioProcessor::new();
        let very_quiet = processor
            .preprocess_decoded(&DecodedAudio {
                samples: seeded_signal(0.001),
                sample_rate: 16_000,
            })
            .unwrap();
        let less_quiet = processor
            .preprocess_decoded(&DecodedAudio {
                samples: seeded_signal(0.004),
                sample_rate: 16_000,
            })
            .unwrap();

        assert_eq!(very_quiet, less_quiet);
    }

    #[test]
    fn rms_floor_leaves_silence_and_loud_audio_unchanged() {
        let processor = InklingAudioProcessor::new();
        let mut without_floor = processor.clone();
        without_floor.params.audio_rms_norm_floor = 0.0;

        for samples in [vec![0.0; 1600], seeded_signal(1.0)] {
            let with_floor = processor
                .preprocess_decoded(&DecodedAudio {
                    samples: samples.clone(),
                    sample_rate: 16_000,
                })
                .unwrap();
            let without_floor = without_floor
                .preprocess_decoded(&DecodedAudio {
                    samples,
                    sample_rate: 16_000,
                })
                .unwrap();
            assert_eq!(with_floor, without_floor);
        }
    }

    #[test]
    fn rejects_invalid_rms_floor() {
        let mut processor = InklingAudioProcessor::new();
        processor.params.audio_rms_norm_floor = -0.01;
        let error = processor
            .preprocess_decoded(&DecodedAudio {
                samples: vec![0.1; 800],
                sample_rate: 16_000,
            })
            .unwrap_err();
        assert!(error.to_string().contains("audio_rms_norm_floor"));
    }

    /// The computation this file replaced, kept here as the bit-identity
    /// reference: one clip, a frame at a time, the tables built per call.
    fn frame_by_frame_reference(
        processor: &InklingAudioProcessor,
        decoded: &DecodedAudio,
    ) -> Result<Array2<f32>, TransformError> {
        let mut samples = if decoded.sample_rate == processor.params.sample_rate {
            decoded.samples.clone()
        } else {
            bandlimited_resample(
                &decoded.samples,
                decoded.sample_rate,
                processor.params.sample_rate,
            )?
        };
        normalize_audio_rms(&mut samples, processor.params.audio_rms_norm_floor)?;
        frame_by_frame_dmel_bins(&samples, &processor.params)
    }

    fn frame_by_frame_dmel_bins(
        samples: &[f32],
        params: &InklingAudioEncoderParams,
    ) -> Result<Array2<f32>, TransformError> {
        if params.n_mels == 0 || params.num_dmel_bins == 0 {
            return Err(TransformError::ShapeError(
                "n_mels and num_dmel_bins must be positive".to_string(),
            ));
        }

        let hop_length = params.hop_length()?;
        let window_size = params.window_size()?;
        let n_fft = params.n_fft.unwrap_or(window_size);
        if n_fft < window_size {
            return Err(TransformError::ShapeError(format!(
                "n_fft ({n_fft}) must be greater than or equal to window_size ({window_size})"
            )));
        }
        if samples.is_empty() {
            return Err(TransformError::ShapeError(
                "audio preprocessing requires at least one sample".to_string(),
            ));
        }

        let right_pad = samples.len().div_ceil(hop_length) * hop_length - samples.len();
        let left_pad = n_fft.saturating_sub(hop_length);
        let padded_len = left_pad + samples.len() + right_pad;
        let mut padded = vec![0.0_f32; padded_len];
        padded[left_pad..left_pad + samples.len()].copy_from_slice(samples);

        let frame_count = (padded_len - n_fft) / hop_length + 1;
        let fft_bins = n_fft / 2 + 1;
        let window = hann_window(window_size);
        let mel_basis = mel_basis(params.sample_rate, n_fft, params.n_mels);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let mut buffer = vec![Complex32::new(0.0, 0.0); n_fft];
        let mut magnitudes = vec![0.0_f32; fft_bins * frame_count];

        for frame in 0..frame_count {
            buffer.fill(Complex32::new(0.0, 0.0));
            let start = frame * hop_length;
            for i in 0..window_size {
                buffer[i].re = padded[start + i] * window[i];
            }
            fft.process(&mut buffer);
            for bin in 0..fft_bins {
                let value = buffer[bin];
                magnitudes[bin * frame_count + frame] =
                    (value.re.mul_add(value.re, value.im * value.im))
                        .max(1e-10)
                        .sqrt();
            }
        }

        let mut output = Vec::with_capacity(frame_count * params.n_mels);
        for frame in 0..frame_count {
            for mel in 0..params.n_mels {
                let basis_row = &mel_basis[mel * fft_bins..(mel + 1) * fft_bins];
                let mut value = 0.0_f32;
                for bin in 0..fft_bins {
                    value += basis_row[bin] * magnitudes[bin * frame_count + frame];
                }
                let log_mel = f64::from(value.max(1e-10).log10())
                    .clamp(params.dmel_min_value, params.dmel_max_value);
                output.push(quantize_dmel(log_mel, params) as f32);
            }
        }

        Array2::from_shape_vec((frame_count, params.n_mels), output).map_err(|e| {
            TransformError::ShapeError(format!(
                "failed to create Inkling dMel bins [{frame_count}, {}]: {e}",
                params.n_mels
            ))
        })
    }

    /// Clips of the lengths that exercise the frame geometry: a hop, a frame,
    /// the chunk boundary (32 frames of 800 samples), either side of it, and
    /// a thirty-second clip; quiet (raised to the floor), ordinary and loud.
    fn waveform_corpus() -> Vec<(String, DecodedAudio)> {
        test_support::waveform_corpus(&[
            1, 799, 800, 801, 1600, 16_000, 24_800, 25_600, 25_601, 26_400, 480_000,
        ])
    }

    fn assert_same_bits(name: &str, actual: &Array2<f32>, expected: &Array2<f32>) {
        assert_eq!(actual.shape(), expected.shape(), "{name}: shape");
        for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
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
        let processor = InklingAudioProcessor::new();
        for (name, decoded) in waveform_corpus() {
            let actual = processor.preprocess_decoded(&decoded).expect(&name);
            let expected = frame_by_frame_reference(&processor, &decoded).expect(&name);
            assert_same_bits(&name, &actual, &expected);
        }
    }

    #[test]
    fn a_batch_equals_each_clip_alone_in_order() {
        let processor = InklingAudioProcessor::new();
        let corpus = waveform_corpus();
        let batch: Vec<&DecodedAudio> = corpus.iter().map(|(_, clip)| clip).collect();
        let inputs = processor.preprocess_decoded_clips(&batch).expect("batch");
        let values = inputs.encoder_input.as_f32();
        let rows = values.as_slice().expect("contiguous");
        let mut offset = 0;
        for ((name, clip), &tokens) in corpus.iter().zip(&inputs.feature_token_counts) {
            let alone = processor.preprocess_decoded(clip).expect(name);
            assert_eq!(alone.shape()[0], tokens, "{name}: tokens");
            let alone = alone.as_slice().expect("contiguous");
            assert_eq!(&rows[offset..offset + alone.len()], alone, "{name}: rows");
            offset += alone.len();
        }
        assert_eq!(offset, rows.len());
    }

    #[test]
    fn a_spectrum_that_overflows_takes_the_full_product_and_keeps_its_bits() {
        let processor = InklingAudioProcessor::new();
        let decoded = DecodedAudio {
            samples: (0..4_000)
                .map(|i| if i % 2 == 0 { 1e30 } else { -1e30 })
                .collect(),
            sample_rate: 16_000,
        };
        let actual = processor
            .preprocess_decoded(&decoded)
            .expect("overflowing clip");
        let expected = frame_by_frame_reference(&processor, &decoded).expect("reference");
        assert_same_bits("overflowing clip", &actual, &expected);
    }

    #[test]
    fn a_processor_clone_shares_the_built_frontend() {
        let processor = InklingAudioProcessor::new();
        let clip = DecodedAudio {
            samples: seeded_signal(0.5),
            sample_rate: 16_000,
        };
        processor.preprocess_decoded(&clip).expect("first clip");
        let clone = processor.clone();
        assert!(Arc::ptr_eq(
            processor.frontend.get().expect("built"),
            clone.frontend.get().expect("shared")
        ));
    }
}
