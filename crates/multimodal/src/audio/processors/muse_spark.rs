//! Muse Spark waveform preprocessing.
//!
//! The family's audio tower consumes mono samples at one fixed rate directly,
//! with no spectrogram front end, and folds a fixed number of consecutive
//! samples into one token. Its reference feature extractor therefore only
//! pads: every clip is right-padded with zeros to a whole number of tokens,
//! and the clip's token count is the padded length over the samples per
//! token. Audio at any other rate is resampled before it reaches the
//! extractor; the gateway's reading of that step is the band-limited sinc
//! filter in [`crate::audio::transforms`], evaluated exactly in integer
//! arithmetic and rounded to `f32` once, since for this family the resampled
//! samples are the features themselves.
//!
//! The processor ships exactly the extractor's rows: each clip's resampled,
//! padded samples as `[samples, 1]`, concatenated across the batch, with the
//! per-clip row counts in [`MUSE_SPARK_SAMPLES_PER_AUDIO_KEY`] so the batch
//! can be split back per clip.

use std::sync::Arc;

use ndarray::Array2;
use serde_json::Value;
use thiserror::Error;

use crate::{
    audio::{transforms::bandlimited_resample_exact, AudioPreProcessor, DecodedAudio},
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
    vision::PreProcessorConfig,
};

/// Sample rate the family's audio tower was trained on.
pub const MUSE_SPARK_AUDIO_SAMPLE_RATE: usize = 24_000;
/// Samples the tower folds into one audio token, the product of its
/// `audio_config.patch_sizes`.
pub const MUSE_SPARK_AUDIO_SAMPLES_PER_TOKEN: usize = 1_920;
/// Side tensor naming each clip's row count in the flat batch.
pub const MUSE_SPARK_SAMPLES_PER_AUDIO_KEY: &str = "samples_per_audio";

/// A Muse Spark audio configuration the processor cannot serve.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MuseSparkAudioConfigError {
    /// A field is present but not a usable value.
    #[error("config field '{field}' is present but unusable: {reason}")]
    Invalid { field: String, reason: String },
}

/// The extractor parameters: the rate the tower expects, the samples per
/// token and the value padding is written with.
#[derive(Debug, Clone, PartialEq)]
pub struct MuseSparkAudioParams {
    pub sample_rate: usize,
    pub samples_per_token: usize,
    pub padding_value: f32,
}

impl Default for MuseSparkAudioParams {
    fn default() -> Self {
        Self {
            sample_rate: MUSE_SPARK_AUDIO_SAMPLE_RATE,
            samples_per_token: MUSE_SPARK_AUDIO_SAMPLES_PER_TOKEN,
            padding_value: 0.0,
        }
    }
}

impl MuseSparkAudioParams {
    /// The parameters a checkpoint declares.
    ///
    /// The feature extractor section of the processor config carries
    /// `sampling_rate`, `samples_per_token` and `padding_value`; the model
    /// config's `audio_config.patch_sizes` multiply to the samples per token
    /// as well, since the tower folds exactly that many. A checkpoint that
    /// states both must agree, or its audio cannot be encoded as it asks.
    pub fn from_configs(
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> Result<Self, MuseSparkAudioConfigError> {
        let mut params = Self::default();

        if let Some(value) = preprocessor_config.extra.get("sampling_rate") {
            params.sample_rate = positive_usize(value, "sampling_rate")?;
        }
        if let Some(value) = preprocessor_config.extra.get("padding_value") {
            params.padding_value = value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                MuseSparkAudioConfigError::Invalid {
                    field: "padding_value".to_string(),
                    reason: "expected a finite number".to_string(),
                }
            })? as f32;
        }

        let declared = preprocessor_config
            .extra
            .get("samples_per_token")
            .map(|value| positive_usize(value, "samples_per_token"))
            .transpose()?;
        let folded = Self::tower_samples_per_token(model_config)?;
        params.samples_per_token = match (declared, folded) {
            (Some(declared), Some(folded)) if declared != folded => {
                return Err(MuseSparkAudioConfigError::Invalid {
                    field: "samples_per_token".to_string(),
                    reason: format!(
                        "{declared} differs from the {folded} samples audio_config.patch_sizes fold into one token"
                    ),
                });
            }
            (Some(declared), _) => declared,
            (None, Some(folded)) => folded,
            (None, None) => MUSE_SPARK_AUDIO_SAMPLES_PER_TOKEN,
        };

        Ok(params)
    }

    /// The samples the tower folds into one token, when the model config
    /// declares the tower's patch sizes.
    fn tower_samples_per_token(
        model_config: &Value,
    ) -> Result<Option<usize>, MuseSparkAudioConfigError> {
        let Some(patch_sizes) = model_config
            .get("audio_config")
            .and_then(|audio| audio.get("patch_sizes"))
        else {
            return Ok(None);
        };
        let invalid = |reason: &str| MuseSparkAudioConfigError::Invalid {
            field: "audio_config.patch_sizes".to_string(),
            reason: reason.to_string(),
        };
        let sizes = patch_sizes
            .as_array()
            .ok_or_else(|| invalid("expected a list of positive integers"))?;
        if sizes.is_empty() {
            return Err(invalid("expected at least one patch size"));
        }
        let mut product = 1usize;
        for size in sizes {
            let size = positive_usize(size, "audio_config.patch_sizes")?;
            product = product
                .checked_mul(size)
                .ok_or_else(|| invalid("product overflows"))?;
        }
        Ok(Some(product))
    }
}

fn positive_usize(value: &Value, field: &str) -> Result<usize, MuseSparkAudioConfigError> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| MuseSparkAudioConfigError::Invalid {
            field: field.to_string(),
            reason: "expected a positive integer".to_string(),
        })
}

/// Resample each clip to the tower's rate, pad it to whole tokens and count
/// its tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct MuseSparkAudioProcessor {
    params: MuseSparkAudioParams,
}

impl Default for MuseSparkAudioProcessor {
    fn default() -> Self {
        Self::new(MuseSparkAudioParams::default())
    }
}

impl MuseSparkAudioProcessor {
    pub fn new(params: MuseSparkAudioParams) -> Self {
        Self { params }
    }

    /// The processor a checkpoint's configs describe.
    pub fn from_configs(
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> Result<Self, MuseSparkAudioConfigError> {
        MuseSparkAudioParams::from_configs(model_config, preprocessor_config).map(Self::new)
    }

    pub fn params(&self) -> &MuseSparkAudioParams {
        &self.params
    }

    /// Tokens the tower emits for `samples` samples at its rate: the clip
    /// padded up to whole tokens.
    pub fn token_count(&self, samples: usize) -> usize {
        samples.div_ceil(self.params.samples_per_token)
    }

    /// Preprocess already-decoded clips.
    ///
    /// The encoder input is `[total padded samples, 1]` with the clips in
    /// order, each padded to its own whole tokens; `samples_per_audio` holds
    /// the padded length of each clip.
    pub fn preprocess_decoded(
        &self,
        clips: &[DecodedAudio],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if clips.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let samples_per_token = self.params.samples_per_token;
        let mut samples: Vec<f32> = Vec::new();
        let mut token_counts = Vec::with_capacity(clips.len());
        let mut item_sizes = Vec::with_capacity(clips.len());
        let mut row_counts = Vec::with_capacity(clips.len());
        for (clip_index, clip) in clips.iter().enumerate() {
            if clip.sample_rate == 0 {
                return Err(TransformError::ShapeError(format!(
                    "audio clip at batch index {clip_index} has no sample rate"
                )));
            }
            if clip.samples.is_empty() {
                return Err(TransformError::ShapeError(format!(
                    "audio clip at batch index {clip_index} contains no samples"
                )));
            }
            if clip.samples.iter().any(|sample| !sample.is_finite()) {
                return Err(TransformError::ShapeError(format!(
                    "audio clip at batch index {clip_index} contains a non-finite sample"
                )));
            }
            let start = samples.len();
            if clip.sample_rate == self.params.sample_rate {
                samples.extend_from_slice(&clip.samples);
            } else {
                samples.extend(bandlimited_resample_exact(
                    &clip.samples,
                    clip.sample_rate,
                    self.params.sample_rate,
                )?);
            }
            let resampled = samples.len() - start;
            if resampled == 0 {
                return Err(TransformError::ShapeError(format!(
                    "audio clip at batch index {clip_index} is empty after resampling"
                )));
            }
            let tokens = self.token_count(resampled);
            let padded = tokens * samples_per_token;
            samples.resize(start + padded, self.params.padding_value);
            let rows = u32::try_from(padded).map_err(|_| {
                TransformError::ShapeError(format!(
                    "audio clip at batch index {clip_index} has too many samples ({padded})"
                ))
            })?;
            token_counts.push(tokens);
            item_sizes.push((rows, 1));
            row_counts.push(padded as i64);
        }
        let rows = samples.len();
        let encoder_input = Array2::from_shape_vec((rows, 1), samples).map_err(|error| {
            TransformError::ShapeError(format!("invalid audio sample buffer: {error}"))
        })?;
        Ok(
            PreprocessedEncoderInputs::new(encoder_input, token_counts, item_sizes).with_extra(
                MUSE_SPARK_SAMPLES_PER_AUDIO_KEY,
                ModelSpecificValue::int_1d(row_counts),
            ),
        )
    }
}

impl AudioPreProcessor for MuseSparkAudioProcessor {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let decoded: Vec<DecodedAudio> = clips.iter().map(|clip| clip.decoded().clone()).collect();
        self.preprocess_decoded(&decoded)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn decoded(samples: usize, sample_rate: usize) -> DecodedAudio {
        DecodedAudio {
            samples: (0..samples).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect(),
            sample_rate,
        }
    }

    fn rows(output: &PreprocessedEncoderInputs) -> Vec<i64> {
        match output.model_specific.get(MUSE_SPARK_SAMPLES_PER_AUDIO_KEY) {
            Some(ModelSpecificValue::IntTensor { data, shape }) => {
                assert_eq!(shape, &vec![data.len()]);
                data.clone()
            }
            other => panic!("unexpected samples_per_audio: {other:?}"),
        }
    }

    #[test]
    fn native_rate_clips_pass_through_padded_to_whole_tokens() {
        let output = MuseSparkAudioProcessor::default()
            .preprocess_decoded(&[decoded(1_920 * 3 + 5, 24_000), decoded(1_920, 24_000)])
            .unwrap();
        assert_eq!(output.feature_token_counts, vec![4, 1]);
        assert_eq!(output.encoder_input.shape(), &[1_920 * 5, 1]);
        assert_eq!(output.item_sizes, vec![(1_920 * 4, 1), (1_920, 1)]);
        assert_eq!(rows(&output), vec![1_920 * 4, 1_920]);
        let flat = output.encoder_input.flat_f32();
        let first = decoded(1_920 * 3 + 5, 24_000).samples;
        assert_eq!(&flat[..first.len()], &first[..]);
        assert!(flat[first.len()..1_920 * 4].iter().all(|&s| s == 0.0));
        let second = decoded(1_920, 24_000).samples;
        assert_eq!(&flat[1_920 * 4..], &second[..]);
    }

    #[test]
    fn one_sample_is_one_token_of_silence_after_it() {
        let output = MuseSparkAudioProcessor::default()
            .preprocess_decoded(&[DecodedAudio {
                samples: vec![0.25],
                sample_rate: 24_000,
            }])
            .unwrap();
        assert_eq!(output.feature_token_counts, vec![1]);
        let flat = output.encoder_input.flat_f32();
        assert_eq!(flat.len(), 1_920);
        assert_eq!(flat[0], 0.25);
        assert!(flat[1..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn other_rates_are_resampled_before_counting_tokens() {
        let output = MuseSparkAudioProcessor::default()
            .preprocess_decoded(&[decoded(16_000, 16_000)])
            .unwrap();
        // 24,000 samples after resampling: 12.5 tokens, padded to 13.
        assert_eq!(output.feature_token_counts, vec![13]);
        assert_eq!(output.encoder_input.shape(), &[13 * 1_920, 1]);
        assert_eq!(rows(&output), vec![13 * 1_920]);
    }

    #[test]
    fn empty_and_non_finite_clips_are_rejected() {
        let processor = MuseSparkAudioProcessor::default();
        assert!(matches!(
            processor.preprocess_decoded(&[]),
            Err(TransformError::EmptyBatch)
        ));
        assert!(processor.preprocess_decoded(&[decoded(0, 24_000)]).is_err());
        assert!(processor.preprocess_decoded(&[decoded(10, 0)]).is_err());
        let mut clip = decoded(100, 24_000);
        clip.samples[5] = f32::NAN;
        assert!(processor.preprocess_decoded(&[clip]).is_err());
    }

    #[test]
    fn params_come_from_the_extractor_section_and_the_tower_geometry() {
        let model = json!({"audio_config": {"patch_sizes": [16, 6, 5, 4]}});
        let extractor = PreProcessorConfig::from_json(
            r#"{"feature_extractor_type": "MuseSparkFeatureExtractor", "feature_size": 1,
                "padding_side": "right", "padding_value": 0.0, "return_attention_mask": true,
                "samples_per_token": 1920, "sampling_rate": 24000}"#,
        )
        .unwrap();
        assert_eq!(
            MuseSparkAudioParams::from_configs(&model, &extractor).unwrap(),
            MuseSparkAudioParams::default()
        );

        // The tower geometry alone fixes the samples per token.
        let from_tower =
            MuseSparkAudioParams::from_configs(&model, &PreProcessorConfig::default()).unwrap();
        assert_eq!(from_tower.samples_per_token, 1_920);
        assert_eq!(from_tower.sample_rate, 24_000);

        // Another geometry is followed, not overridden by the family default.
        let other = json!({"audio_config": {"patch_sizes": [8, 5, 4]}});
        let processor =
            MuseSparkAudioProcessor::from_configs(&other, &PreProcessorConfig::default()).unwrap();
        assert_eq!(processor.params().samples_per_token, 160);
        assert_eq!(processor.token_count(161), 2);

        // Nothing declared: the family's constants.
        assert_eq!(
            MuseSparkAudioParams::from_configs(&json!({}), &PreProcessorConfig::default()).unwrap(),
            MuseSparkAudioParams::default()
        );
    }

    #[test]
    fn disagreeing_or_unusable_geometry_is_an_error_naming_the_field() {
        let model = json!({"audio_config": {"patch_sizes": [16, 6, 5, 4]}});
        let extractor = PreProcessorConfig::from_json(r#"{"samples_per_token": 1600}"#).unwrap();
        assert_eq!(
            MuseSparkAudioParams::from_configs(&model, &extractor).unwrap_err(),
            MuseSparkAudioConfigError::Invalid {
                field: "samples_per_token".to_string(),
                reason: "1600 differs from the 1920 samples audio_config.patch_sizes fold into one token"
                    .to_string(),
            }
        );
        let zero_rate = PreProcessorConfig::from_json(r#"{"sampling_rate": 0}"#).unwrap();
        assert!(matches!(
            MuseSparkAudioParams::from_configs(&model, &zero_rate),
            Err(MuseSparkAudioConfigError::Invalid { field, .. }) if field == "sampling_rate"
        ));
        let bad_sizes = json!({"audio_config": {"patch_sizes": [16, 0]}});
        assert!(matches!(
            MuseSparkAudioParams::from_configs(&bad_sizes, &PreProcessorConfig::default()),
            Err(MuseSparkAudioConfigError::Invalid { field, .. }) if field == "audio_config.patch_sizes"
        ));
        let empty_sizes = json!({"audio_config": {"patch_sizes": []}});
        assert!(
            MuseSparkAudioParams::from_configs(&empty_sizes, &PreProcessorConfig::default())
                .is_err()
        );
    }
}
