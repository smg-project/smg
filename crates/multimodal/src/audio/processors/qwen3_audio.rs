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
    pub fn from_configs(model_config: &Value, preprocessor_config: &PreProcessorConfig) -> Self {
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

    pub fn with_params(params: Qwen3AudioParams) -> Self {
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

    /// The log-mel features of a batch of decoded clips. The clips are borrowed: a clip at the
    /// frontend's sample rate is copied once, into the batch's padded buffer, and only a clip at
    /// another rate is resampled into a buffer of its own first.
    pub fn preprocess_decoded_clips(
        &self,
        clips: &[&DecodedAudio],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.validate_params()?;
        if clips.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

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
            if clip.samples.iter().any(|sample| !sample.is_finite()) {
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
        let clip_values = self.params.n_mels.checked_mul(max_frames).ok_or_else(|| {
            TransformError::ShapeError("Qwen3 audio feature size overflow".to_string())
        })?;
        let feature_values = batch_size.checked_mul(clip_values).ok_or_else(|| {
            TransformError::ShapeError("Qwen3 audio feature size overflow".to_string())
        })?;
        let frontend = self
            .frontend
            .get_or_init(|| Arc::new(MelFrontend::build(&self.params)));
        let mut scratch = MelScratch::new(frontend, &self.params, max_samples);
        let mut all_features = vec![0.0_f32; feature_values];
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
            )?;

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

/// The frames of a clip run through the FFT so many at a time: the chunk's complex frames stay
/// in cache (32 frames of 400 bins are 100 KiB) and one FFT call covers them all.
const FRAMES_PER_CHUNK: usize = 32;

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

/// The buffers one batch reuses for every clip: the padded waveform, a chunk of complex frames,
/// the FFT's own scratch and the chunk's power spectra.
struct MelScratch {
    padded: Vec<f32>,
    frames: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    power: Vec<f32>,
}

impl MelScratch {
    fn new(frontend: &MelFrontend, params: &Qwen3AudioParams, max_samples: usize) -> Self {
        let fft_bins = params.n_fft / 2 + 1;
        Self {
            padded: Vec::with_capacity(max_samples + 2 * (params.n_fft / 2)),
            frames: vec![Complex32::new(0.0, 0.0); FRAMES_PER_CHUNK * params.n_fft],
            fft_scratch: vec![Complex32::new(0.0, 0.0); frontend.fft.get_inplace_scratch_len()],
            power: vec![0.0_f32; FRAMES_PER_CHUNK * fft_bins],
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
    /// bits do not move. The clip is extended to `padded_len` (the batch's longest) with the
    /// padding value before the centre reflection, as the reference pads a batch.
    fn log_mel(
        &self,
        samples: &[f32],
        padded_len: usize,
        frame_count: usize,
        params: &Qwen3AudioParams,
        scratch: &mut MelScratch,
        out: &mut [f32],
    ) -> Result<(), TransformError> {
        let n_fft = params.n_fft;
        let fft_bins = n_fft / 2 + 1;
        reflect_pad_into(
            &mut scratch.padded,
            samples,
            padded_len,
            params.padding_value,
            n_fft / 2,
        );

        let mut frame = 0;
        while frame < frame_count {
            let count = FRAMES_PER_CHUNK.min(frame_count - frame);
            for k in 0..count {
                let start = (frame + k) * params.hop_length;
                let frame_samples = scratch.padded.get(start..start + n_fft).ok_or_else(|| {
                    TransformError::ShapeError(format!(
                        "Qwen3 audio STFT frame {} lies outside padded waveform",
                        frame + k
                    ))
                })?;
                let buffer = &mut scratch.frames[k * n_fft..(k + 1) * n_fft];
                for (slot, (&sample, &weight)) in buffer
                    .iter_mut()
                    .zip(frame_samples.iter().zip(&self.window))
                {
                    *slot = Complex32::new(sample * weight, 0.0);
                }
            }
            self.fft.process_with_scratch(
                &mut scratch.frames[..count * n_fft],
                &mut scratch.fft_scratch,
            );

            // The power term of each bin once per frame; `mul_add`, as the product reads it,
            // not `re * re + im * im`, which rounds twice and differs on real spectra (the
            // sibling frontends hoist the unfused form; this one must not copy it).
            for k in 0..count {
                let spectrum = &scratch.frames[k * n_fft..k * n_fft + fft_bins];
                let power = &mut scratch.power[k * fft_bins..(k + 1) * fft_bins];
                for (value, bin) in power.iter_mut().zip(spectrum) {
                    *value = bin.re.mul_add(bin.re, bin.im * bin.im);
                }
            }

            // The mel product over each filter's band: a zero weight adds nothing to a finite
            // power, so the band's sum is the full row's sum, bit for bit. Where a power
            // overflowed, a zero weight times infinity is NaN and the full row keeps it.
            for k in 0..count {
                let power = &scratch.power[k * fft_bins..(k + 1) * fft_bins];
                let finite = power.iter().all(|value| value.is_finite());
                for mel in 0..params.n_mels {
                    let (first, last) = if finite {
                        self.bands[mel]
                    } else {
                        (0, fft_bins)
                    };
                    let weights = &self.filters[mel * fft_bins + first..mel * fft_bins + last];
                    let mut value = 0.0_f32;
                    for (weight, power) in weights.iter().zip(&power[first..last]) {
                        value = weight.mul_add(*power, value);
                    }
                    out[mel * frame_count + frame + k] = value.max(1e-10).log10();
                }
            }
            frame += count;
        }

        let peak = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let floor = peak - 8.0;
        for value in out.iter_mut() {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
        Ok(())
    }
}

/// `out` becomes `samples` extended to `len` with `padding_value`, then reflected `padding`
/// samples at both ends (numpy's `reflect` mode, which the reference's centred STFT uses). The
/// buffer's capacity is kept across clips.
fn reflect_pad_into(
    out: &mut Vec<f32>,
    samples: &[f32],
    len: usize,
    padding_value: f32,
    padding: usize,
) {
    out.clear();
    let value_at = |index: usize| samples.get(index).copied().unwrap_or(padding_value);
    if padding == 0 {
        out.extend((0..len).map(value_at));
        return;
    }
    if len == 1 {
        out.resize(1 + 2 * padding, value_at(0));
        return;
    }
    out.extend(
        (-(padding as isize)..(len + padding) as isize)
            .map(|position| value_at(reflect_index(position, len))),
    );
}

fn reflect_index(mut index: isize, len: usize) -> usize {
    let last = len as isize - 1;
    while index < 0 || index > last {
        if index < 0 {
            index = -index;
        }
        if index > last {
            index = 2 * last - index;
        }
    }
    index as usize
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
        let mut out = Vec::new();
        reflect_pad_into(&mut out, &[1.0, 2.0, 3.0], 3, 0.0, 2);
        assert_eq!(out, vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
        reflect_pad_into(&mut out, &[2.0], 1, 0.0, 2);
        assert_eq!(out, vec![2.0; 5]);
        // A clip shorter than the batch's longest is extended with the padding value first, so
        // the reflection at its end mirrors the padding.
        reflect_pad_into(&mut out, &[1.0, 2.0], 4, 0.0, 2);
        assert_eq!(out, vec![0.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        reflect_pad_into(&mut out, &[1.0, 2.0], 3, 0.0, 0);
        assert_eq!(out, vec![1.0, 2.0, 0.0]);
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
        fn reflect_pad(samples: &[f32], padding: usize) -> Vec<f32> {
            if padding == 0 {
                return samples.to_vec();
            }
            if samples.len() == 1 {
                return vec![samples[0]; samples.len() + 2 * padding];
            }
            let mut padded = Vec::with_capacity(samples.len() + 2 * padding);
            for position in -(padding as isize)..(samples.len() + padding) as isize {
                padded.push(samples[reflect_index(position, samples.len())]);
            }
            padded
        }
        let mut waveform = samples.to_vec();
        waveform.resize(padded_len, params.padding_value);
        let padded = reflect_pad(&waveform, params.n_fft / 2);
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

    /// Deterministic waveforms over the paths a numerical change would move: a tone, noise,
    /// silence, an impulse, a loud square wave, a very quiet tone, and lengths on both sides of
    /// a chunk boundary (31, 32, 33 frames), a single frame and a part frame.
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
        for frames in [1, 31, 32, 33, 65] {
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

        // And an ordinary clip takes the band product throughout.
        let (_, ordinary) = waveform_corpus().swap_remove(1);
        assert!(
            !a_power_term_overflows(&ordinary, &params),
            "an ordinary clip should take the band product"
        );
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
