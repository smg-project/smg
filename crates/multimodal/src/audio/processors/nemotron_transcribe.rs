//! Native Nemotron-3.5-Transcribe waveform inputs. The model owns the mel
//! frontend; the host only resamples, minimally pads and counts embeddings.

use std::sync::Arc;

use ndarray::Array2;
use serde_json::Value;
use thiserror::Error;

use crate::{
    audio::{transforms::bandlimited_resample, AudioPreProcessor, DecodedAudio},
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
};

pub const NEMOTRON_AUDIO_LENGTH_KEY: &str = "audio_signal_length";

/// Maximum bytes in the dense, right-padded f32 waveform batch (256 MiB).
/// Checked before resampling or allocating the output tensor. This is a
/// processor output limit, independent of the decoder's per-clip byte limit;
/// decoded inputs and one clip's resampling scratch are additional memory.
pub const NEMOTRON_MAX_PADDED_AUDIO_BYTES: usize = 256 * 1024 * 1024;

/// A required geometry field is absent, or a supplied value is invalid.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NemotronTranscribeConfigError {
    #[error("missing {field}")]
    Missing { field: String },
    #[error("invalid {field}: {reason}")]
    Invalid { field: String, reason: String },
}

/// Geometry consumed by vLLM's native audio tower, resolved from either its
/// normalized config or the published NeMo checkpoint's perception config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronTranscribeAudioParams {
    pub sample_rate: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub subsampling_factor: usize,
}

impl NemotronTranscribeAudioParams {
    pub fn from_config(config: &Value) -> Result<Self, NemotronTranscribeConfigError> {
        let invalid = |field: &str, reason: &str| NemotronTranscribeConfigError::Invalid {
            field: field.to_string(),
            reason: reason.to_string(),
        };
        let positive = |value: Option<&Value>, field: &str| {
            let value = value.ok_or_else(|| NemotronTranscribeConfigError::Missing {
                field: field.to_string(),
            })?;
            value
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|v| *v > 0)
                .ok_or_else(|| invalid(field, "expected a positive integer"))
        };
        // vLLM re-resolves perception when present, even if audio_config exists.
        let (sample_rate, hop_length, n_fft, factor) = if config
            .get("perception")
            .is_some_and(|p| !p.is_null() && p.as_object().is_none_or(|p| !p.is_empty()))
        {
            let pre = &config["perception"]["preprocessor"];
            let default_rate = Value::from(16_000);
            let sample_rate = positive(
                pre.get("sample_rate").or(Some(&default_rate)),
                "perception.preprocessor.sample_rate",
            )?;
            let stride = pre
                .get("window_stride")
                .map_or(Some(0.01), Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| {
                    invalid(
                        "perception.preprocessor.window_stride",
                        "expected positive seconds",
                    )
                })?;
            let hop = stride * sample_rate as f64;
            if hop < 1.0 || hop >= usize::MAX as f64 {
                return Err(invalid(
                    "perception.preprocessor.window_stride",
                    "hop length is out of range",
                ));
            }
            let n_fft = if pre
                .get("n_fft")
                .is_none_or(|v| v.is_null() || v.as_u64() == Some(0))
            {
                let window = pre
                    .get("window_size")
                    .map_or(Some(0.02), Value::as_f64)
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .ok_or_else(|| {
                        invalid(
                            "perception.preprocessor.window_size",
                            "expected positive seconds",
                        )
                    })?;
                let samples = window * sample_rate as f64;
                if samples < 1.0 || samples >= usize::MAX as f64 {
                    return Err(invalid(
                        "perception.preprocessor.window_size",
                        "window is out of range",
                    ));
                }
                (samples as usize)
                    .checked_next_power_of_two()
                    .ok_or_else(|| {
                        invalid("perception.preprocessor.window_size", "FFT size overflows")
                    })?
            } else {
                positive(pre.get("n_fft"), "perception.preprocessor.n_fft")?
            };
            let default_factor = Value::from(4);
            let factor = positive(
                config
                    .pointer("/pe_encoder_config/asr_encoder_cfg/subsampling_factor")
                    .or(Some(&default_factor)),
                "pe_encoder_config.asr_encoder_cfg.subsampling_factor",
            )?;
            (sample_rate, hop as usize, n_fft, factor)
        } else {
            let pre = &config["audio_config"]["preprocessor"];
            (
                positive(
                    pre.get("sample_rate"),
                    "audio_config.preprocessor.sample_rate",
                )?,
                positive(
                    pre.get("hop_length"),
                    "audio_config.preprocessor.hop_length",
                )?,
                positive(pre.get("n_fft"), "audio_config.preprocessor.n_fft")?,
                positive(
                    config.pointer("/audio_config/asr_encoder/subsampling_factor"),
                    "audio_config.asr_encoder.subsampling_factor",
                )?,
            )
        };
        // The published native family consumes 16 kHz mono. Refuse other
        // towers instead of silently treating an unrelated SpeechLM as this model.
        if sample_rate != 16_000 {
            return Err(invalid(
                "sample_rate",
                "Nemotron Transcribe requires 16000 Hz",
            ));
        }
        if n_fft % 2 != 0 {
            return Err(invalid("n_fft", "expected an even FFT size"));
        }
        Ok(Self {
            sample_rate,
            hop_length,
            n_fft,
            subsampling_factor: factor,
        })
    }

    /// Native `_num_audio_embeddings`: STFT frame floor followed by the
    /// feature-stacking ceiling. Minimal padding supplies at least one frame.
    pub fn token_count(&self, samples: usize) -> usize {
        (samples.max(self.hop_length) / self.hop_length).div_ceil(self.subsampling_factor)
    }
}

#[derive(Debug, Clone)]
pub struct NemotronTranscribeAudioProcessor {
    params: NemotronTranscribeAudioParams,
}

impl NemotronTranscribeAudioProcessor {
    pub fn from_config(config: &Value) -> Result<Self, NemotronTranscribeConfigError> {
        Ok(Self {
            params: NemotronTranscribeAudioParams::from_config(config)?,
        })
    }

    pub fn params(&self) -> &NemotronTranscribeAudioParams {
        &self.params
    }

    /// Batched waveforms right-padded to a common width. The native tower
    /// slices each row to its true post-minimum-pad `audio_signal_length`.
    /// Rejects batches exceeding [`NEMOTRON_MAX_PADDED_AUDIO_BYTES`] before
    /// resampling or allocating their output tensor.
    pub fn preprocess_decoded(
        &self,
        clips: &[DecodedAudio],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_refs(clips.iter())
    }

    fn preprocess_refs<'a>(
        &self,
        clips: impl ExactSizeIterator<Item = &'a DecodedAudio> + Clone,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_refs_with_budget(clips, NEMOTRON_MAX_PADDED_AUDIO_BYTES)
    }

    fn preprocess_refs_with_budget<'a>(
        &self,
        clips: impl ExactSizeIterator<Item = &'a DecodedAudio> + Clone,
        max_bytes: usize,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let batch_size = clips.len();
        if batch_size == 0 {
            return Err(TransformError::EmptyBatch);
        }
        // Match the resampler's ceil(input_samples * target_rate / source_rate).
        // Inspect the entire batch before allocating or resampling any clip:
        // a short clip beside a long one still consumes a full padded row.
        let mut width = 0;
        for (index, clip) in clips.clone().enumerate() {
            if clip.sample_rate == 0
                || clip.samples.is_empty()
                || clip.samples.iter().any(|s| !s.is_finite())
            {
                return Err(TransformError::ShapeError(format!("invalid audio clip at batch index {index}: expected finite samples and a positive sample rate")));
            }
            let length = resampled_length(
                clip.samples.len(),
                clip.sample_rate,
                self.params.sample_rate,
            )?
            .max(self.params.hop_length);
            u32::try_from(length)
                .map_err(|_| TransformError::ShapeError("audio length exceeds u32".to_string()))?;
            width = width.max(length);
        }
        let elements = padded_batch_elements(batch_size, width, max_bytes)?;
        let mut samples = reserved_vec(elements, "padded audio batch")?;
        samples.resize(elements, 0.0);
        let mut lengths = reserved_vec(batch_size, "audio lengths")?;
        let mut counts = reserved_vec(batch_size, "audio token counts")?;
        let mut sizes = reserved_vec(batch_size, "audio item sizes")?;
        for (clip, row) in clips.zip(samples.chunks_exact_mut(width)) {
            // Borrow native-rate clips; keep at most one resampled clip alive.
            let resampled;
            let waveform = if clip.sample_rate == self.params.sample_rate {
                &clip.samples
            } else {
                resampled =
                    bandlimited_resample(&clip.samples, clip.sample_rate, self.params.sample_rate)?;
                &resampled
            };
            let length = waveform.len().max(self.params.hop_length);
            let size = u32::try_from(length)
                .map_err(|_| TransformError::ShapeError("audio length exceeds u32".to_string()))?;
            // Keep a mismatch in the shared resampler's sizing a typed error.
            let target = row.get_mut(..waveform.len()).ok_or_else(|| {
                TransformError::ShapeError("resampled audio exceeds preflight width".to_string())
            })?;
            target.copy_from_slice(waveform);
            lengths.push(i64::from(size));
            sizes.push((size, 1));
            counts.push(self.params.token_count(length));
        }
        let batch = Array2::from_shape_vec((batch_size, width), samples)
            .map_err(|error| TransformError::ShapeError(error.to_string()))?;
        Ok(
            PreprocessedEncoderInputs::new(batch, counts, sizes).with_extra(
                NEMOTRON_AUDIO_LENGTH_KEY,
                ModelSpecificValue::int_1d(lengths),
            ),
        )
    }
}

fn resampled_length(
    samples: usize,
    source_rate: usize,
    target_rate: usize,
) -> Result<usize, TransformError> {
    let length = (samples as u128 * target_rate as u128).div_ceil(source_rate as u128);
    usize::try_from(length)
        .map_err(|_| TransformError::ShapeError("resampled audio length exceeds usize".to_string()))
}

fn padded_batch_elements(
    batch_size: usize,
    width: usize,
    max_bytes: usize,
) -> Result<usize, TransformError> {
    let elements = batch_size.checked_mul(width).ok_or_else(|| {
        TransformError::ShapeError("padded audio batch element count overflow".to_string())
    })?;
    let bytes = elements.checked_mul(size_of::<f32>()).ok_or_else(|| {
        TransformError::ShapeError("padded audio batch byte count overflow".to_string())
    })?;
    if bytes > max_bytes {
        return Err(TransformError::ShapeError(format!(
            "padded audio batch requires {bytes} bytes, exceeding the {max_bytes}-byte output limit"
        )));
    }
    Ok(elements)
}

fn reserved_vec<T>(capacity: usize, label: &str) -> Result<Vec<T>, TransformError> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate {label}: {error}"))
    })?;
    Ok(values)
}

impl AudioPreProcessor for NemotronTranscribeAudioProcessor {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_refs(clips.iter().map(|clip| clip.decoded()))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn config() -> Value {
        json!({"audio_config": {"preprocessor": {"sample_rate":16000,"hop_length":160,"n_fft":512},"asr_encoder":{"subsampling_factor":8}}})
    }
    fn clip(samples: usize, rate: usize) -> DecodedAudio {
        DecodedAudio {
            samples: vec![0.25; samples],
            sample_rate: rate,
        }
    }

    #[test]
    fn native_frame_and_subsampling_boundaries() {
        let params = NemotronTranscribeAudioParams::from_config(&config()).unwrap();
        for (samples, expected) in [
            (0, 1),
            (1, 1),
            (159, 1),
            (160, 1),
            (319, 1),
            (1279, 1),
            (1280, 1),
            (1439, 1),
            (1440, 2),
            (15999, 13),
            (16000, 13),
            (480000, 375),
        ] {
            assert_eq!(params.token_count(samples), expected, "samples={samples}");
        }
        // Even non-power-of-two FFTs cancel in the native center-pad arithmetic.
        let mut varied = config();
        varied["audio_config"]["preprocessor"]["n_fft"] = json!(400);
        assert_eq!(
            NemotronTranscribeAudioParams::from_config(&varied)
                .unwrap()
                .token_count(1440),
            2
        );
    }

    #[test]
    fn public_salm_config_takes_precedence_over_legacy_audio_config() {
        let config = json!({"audio_config": {"model_type":"nemotron3_5_transcribe_audio"},"perception":{"preprocessor":{"sample_rate":16000,"window_stride":0.01,"window_size":0.025,"n_fft":512}},"pe_encoder_config":{"asr_encoder_cfg":{"subsampling_factor":8}}});
        assert_eq!(
            NemotronTranscribeAudioParams::from_config(&config).unwrap(),
            NemotronTranscribeAudioParams {
                sample_rate: 16000,
                hop_length: 160,
                n_fft: 512,
                subsampling_factor: 8
            }
        );
        let mut fallback = config;
        fallback["perception"]["preprocessor"]["n_fft"] = Value::Null;
        assert_eq!(
            NemotronTranscribeAudioParams::from_config(&fallback)
                .unwrap()
                .n_fft,
            512
        );
        fallback["perception"]["preprocessor"]
            .as_object_mut()
            .unwrap()
            .remove("sample_rate");
        fallback["pe_encoder_config"]["asr_encoder_cfg"]
            .as_object_mut()
            .unwrap()
            .remove("subsampling_factor");
        let defaults = NemotronTranscribeAudioParams::from_config(&fallback).unwrap();
        assert_eq!(defaults.sample_rate, 16_000);
        assert_eq!(defaults.subsampling_factor, 4);
    }

    #[test]
    fn batch_padding_preserves_true_lengths_samples_and_counts() {
        let processor = NemotronTranscribeAudioProcessor::from_config(&config()).unwrap();
        let output = processor
            .preprocess_decoded(&[clip(1, 16000), clip(1440, 16000)])
            .unwrap();
        assert_eq!(output.encoder_input.shape(), &[2, 1440]);
        assert_eq!(output.feature_token_counts, [1, 2]);
        assert_eq!(output.item_sizes, [(160, 1), (1440, 1)]);
        match &output.model_specific[NEMOTRON_AUDIO_LENGTH_KEY] {
            ModelSpecificValue::IntTensor { data, shape } => {
                assert_eq!(data, &[160, 1440]);
                assert_eq!(shape, &[2]);
            }
            _ => panic!("expected int64 lengths"),
        }
        let values = output.encoder_input.as_f32();
        assert_eq!(values[[0, 0]], 0.25);
        assert!(values
            .index_axis(ndarray::Axis(0), 0)
            .iter()
            .skip(1)
            .all(|v| *v == 0.0));
        assert!(values
            .index_axis(ndarray::Axis(0), 1)
            .iter()
            .all(|v| *v == 0.25));
    }

    #[test]
    fn padded_batch_budget_counts_padding_and_accepts_exact_limit() {
        let processor = NemotronTranscribeAudioProcessor::from_config(&config()).unwrap();
        let clips = [clip(1, 16000), clip(161, 16000)];
        let required_bytes = 2 * 161 * size_of::<f32>();
        assert!(processor
            .preprocess_refs_with_budget(clips.iter(), required_bytes)
            .is_ok());
        let error = processor
            .preprocess_refs_with_budget(clips.iter(), required_bytes - 1)
            .unwrap_err();
        assert!(error.to_string().contains("output limit"), "{error}");
        // One low-rate sample expands to 16,000 output samples. Budgeting
        // must use the resampled length, before invoking the resampler.
        let clips = [clip(1, 1)];
        let error = processor
            .preprocess_refs_with_budget(clips.iter(), 640)
            .unwrap_err();
        assert!(error.to_string().contains("64000 bytes"), "{error}");
    }

    #[test]
    fn padded_batch_sizing_rejects_overflow_without_allocating() {
        assert!(padded_batch_elements(usize::MAX, 2, usize::MAX).is_err());
        assert!(padded_batch_elements(1, usize::MAX, usize::MAX).is_err());
        assert!(resampled_length(usize::MAX, 1, 16000).is_err());
        assert_eq!(resampled_length(161, 48000, 16000).unwrap(), 54);
        assert_eq!(padded_batch_elements(2, 160, 1280).unwrap(), 320);
    }

    #[test]
    fn equal_length_rows_preserve_distinct_samples() {
        let processor = NemotronTranscribeAudioProcessor::from_config(&config()).unwrap();
        let clips = [
            clip(160, 16000),
            DecodedAudio {
                samples: (0..160).map(|sample| sample as f32 / 160.0).collect(),
                sample_rate: 16000,
            },
        ];
        let output = processor.preprocess_decoded(&clips).unwrap();
        assert_eq!(output.encoder_input.shape(), &[2, 160]);
        for (row, clip) in output.encoder_input.as_f32().outer_iter().zip(&clips) {
            assert_eq!(row.as_slice().unwrap(), clip.samples);
        }
    }

    #[test]
    fn distinguishes_missing_fields_from_invalid_values() {
        for field in ["sample_rate", "hop_length", "n_fft"] {
            let mut config = config();
            config["audio_config"]["preprocessor"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            let path = format!("audio_config.preprocessor.{field}");
            assert_eq!(
                NemotronTranscribeAudioParams::from_config(&config),
                Err(NemotronTranscribeConfigError::Missing {
                    field: path.clone()
                })
            );
            for value in [Value::Null, json!(0), json!("invalid")] {
                config["audio_config"]["preprocessor"][field] = value;
                assert!(matches!(
                    NemotronTranscribeAudioParams::from_config(&config),
                    Err(NemotronTranscribeConfigError::Invalid { field, .. }) if field == path
                ));
            }
        }
        let mut config = config();
        config["audio_config"]["asr_encoder"]
            .as_object_mut()
            .unwrap()
            .remove("subsampling_factor");
        assert_eq!(
            NemotronTranscribeAudioParams::from_config(&config),
            Err(NemotronTranscribeConfigError::Missing {
                field: "audio_config.asr_encoder.subsampling_factor".to_string()
            })
        );
    }

    #[test]
    fn resamples_before_counting_and_rejects_invalid_audio() {
        let processor = NemotronTranscribeAudioProcessor::from_config(&config()).unwrap();
        let output = processor
            .preprocess_decoded(&[clip(8000, 8000), clip(48000, 48000)])
            .unwrap();
        assert_eq!(output.feature_token_counts, [13, 13]);
        match &output.model_specific[NEMOTRON_AUDIO_LENGTH_KEY] {
            ModelSpecificValue::IntTensor { data, shape } => {
                assert_eq!(data, &[16000, 16000]);
                assert_eq!(shape, &[2]);
            }
            _ => panic!("expected int64 lengths"),
        }
        assert!(processor.preprocess_decoded(&[]).is_err());
        for bad in [
            clip(0, 16000),
            clip(4, 0),
            DecodedAudio {
                samples: vec![f32::NAN],
                sample_rate: 16000,
            },
            DecodedAudio {
                samples: vec![f32::INFINITY],
                sample_rate: 16000,
            },
        ] {
            assert!(processor.preprocess_decoded(&[bad]).is_err());
        }
    }

    #[test]
    fn rejects_malformed_geometry() {
        for (path, value) in [
            ("/audio_config/preprocessor/sample_rate", json!(0)),
            ("/audio_config/preprocessor/sample_rate", json!(24000)),
            ("/audio_config/preprocessor/hop_length", json!(0)),
            ("/audio_config/preprocessor/n_fft", json!(511)),
            ("/audio_config/asr_encoder/subsampling_factor", json!(-1)),
            ("/audio_config/asr_encoder/subsampling_factor", json!("8")),
        ] {
            let mut config = config();
            *config.pointer_mut(path).unwrap() = value;
            assert!(
                NemotronTranscribeAudioParams::from_config(&config).is_err(),
                "{path}"
            );
        }
        assert!(NemotronTranscribeAudioParams::from_config(&json!({})).is_err());
    }

    #[tokio::test]
    async fn decoder_downmixes_stereo_before_waveform_preparation() {
        let frames = 160usize;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + frames as u32 * 4).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&64000u32.to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(frames as u32 * 4).to_le_bytes());
        for _ in 0..frames {
            wav.extend_from_slice(&8192i16.to_le_bytes());
            wav.extend_from_slice(&(-8192i16).to_le_bytes());
        }
        let decoded = crate::audio::decode_audio_mono_f32(wav.into())
            .await
            .unwrap();
        assert_eq!(decoded.samples.len(), frames);
        let output = NemotronTranscribeAudioProcessor::from_config(&config())
            .unwrap()
            .preprocess_decoded(&[decoded])
            .unwrap();
        assert!(output.encoder_input.as_f32().iter().all(|s| s.abs() < 1e-6));
    }
}
