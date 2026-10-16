//! Muse Spark: the family's chat template writes one `<|patch|>` per image
//! part, one `<|video|>` per video part and one `<|audio|>` per audio part,
//! and the reference processor rewrites each in place:
//!
//! - an image as `<|image_start|>`, one `<|patch|>` per merged token,
//!   `<|image_end|>`;
//! - a video as `<|vid_start|>`, then per temporal group its stamp (the video
//!   processor's `frame_timestamp_format`, `Time: {timestamp:.2f}s` on the
//!   released checkpoints, of the group's first sampled frame's time) and one
//!   `<|video|>` per merged token of the group, the groups separated by
//!   `<|vid_frame_separator|>` and the last closed by `<|vid_end|>`;
//! - an audio clip as `<|soft_audio_start|>`, one `<|audio|>` per token of
//!   the clip (its padded samples over the samples per token), and
//!   `<|soft_audio_end|>`.
//!
//! The anchor token of each modality is also its fill token, so every
//! replacement declares its feature ranges. Images and frames travel as patch
//! rows with per-item grids (see [`crate::vision::processors::muse_spark`]);
//! audio as the clip's padded samples, `[samples, 1]` rows flat over the
//! batch (see [`crate::audio::processors::muse_spark`]).

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    audio::{
        AudioPreProcessor, MuseSparkAudioConfigError, MuseSparkAudioProcessor,
        MUSE_SPARK_SAMPLES_PER_AUDIO_KEY,
    },
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    media::FrameSampling,
    registry::{
        EncoderInputFormat, MediaItemInfo, MediaPartOrder, ModalityHooks, ModelMetadata,
        ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{
        EncoderFieldLayouts, FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId,
        VideoSamplingInfo,
    },
    vision::{
        processors::muse_spark::{DEFAULT_MERGE_SIZE, DEFAULT_TEMPORAL_PATCH_SIZE},
        PreProcessorConfig,
    },
};

const MODEL_TYPE: &str = "muse_spark";
const ARCHITECTURE: &str = "MuseSparkForConditionalGeneration";
const IMAGE_START: &str = "<|image_start|>";
const IMAGE_END: &str = "<|image_end|>";
const VIDEO_START: &str = "<|vid_start|>";
const VIDEO_END: &str = "<|vid_end|>";
const VIDEO_SEPARATOR: &str = "<|vid_frame_separator|>";
const AUDIO_START: &str = "<|soft_audio_start|>";
const AUDIO_END: &str = "<|soft_audio_end|>";
/// The reference video processor class's frame budget (`num_frames`); the
/// released checkpoints raise it in their processor config.
const DEFAULT_VIDEO_FRAMES: usize = 96;
/// The reference video processor class's sampling rate (`fps`).
const DEFAULT_VIDEO_FPS: f32 = 2.0;
/// The reference video processor class's stamp (`frame_timestamp_format`);
/// the released checkpoints write two decimals.
const DEFAULT_FRAME_TIMESTAMP_FORMAT: &str = "Time: {timestamp:.1f}s";
/// Images per request; raise or lower deployment-wide with
/// `SMG_IMAGE_MAX_COUNT`. The context length is the practical bound.
const MAX_IMAGES_PER_REQUEST: usize = 10;
/// Videos per request (`SMG_VIDEO_MAX_COUNT`).
const MAX_VIDEOS_PER_REQUEST: usize = 1;
/// Audio clips per request (`SMG_AUDIO_MAX_COUNT`).
const MAX_AUDIOS_PER_REQUEST: usize = 10;

pub(super) struct MuseSparkSpec;

impl MuseSparkSpec {
    fn config_token_id(metadata: &ModelMetadata, field: &str) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&[field])
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: field.to_string(),
            })
    }

    /// The vocabulary's spelling of a config-declared token id.
    fn token_for(metadata: &ModelMetadata, field: &str) -> RegistryResult<String> {
        let id = Self::config_token_id(metadata, field)?;
        metadata
            .tokenizer
            .id_to_token(id as u32)
            .ok_or_else(|| ModelRegistryError::TokenNotFound {
                token: format!("{field}:{id}"),
            })
    }

    fn unsupported(modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: "muse_spark",
            modality,
        }
    }

    fn token_counts(preprocessed: &PreprocessedEncoderInputs) -> RegistryResult<&[usize]> {
        if preprocessed.feature_token_counts.is_empty() {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "feature_token_counts".to_string(),
            });
        }
        Ok(&preprocessed.feature_token_counts)
    }

    /// The checkpoint ships the audio tower: an `audio_config` and the token
    /// its prompts carry per audio sample run.
    fn has_audio_tower(config: &Value) -> bool {
        config.get("audio_config").is_some_and(|c| !c.is_null())
            && config.get("audio_token_id").is_some_and(|c| !c.is_null())
    }

    fn audio_config_error(error: MuseSparkAudioConfigError) -> ModelRegistryError {
        match error {
            MuseSparkAudioConfigError::Invalid { field, reason } => {
                ModelRegistryError::InvalidConfigField { field, reason }
            }
        }
    }

    /// One replacement per clip: `<|soft_audio_start|>`, one `<|audio|>` per
    /// token, `<|soft_audio_end|>`, the tokens between them as the feature
    /// span.
    fn audio_replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let placeholder = Self::token_for(metadata, "audio_token_id")?;
        let audio = Self::config_token_id(metadata, "audio_token_id")?;
        let start = metadata.token_id(AUDIO_START)?;
        let end = metadata.token_id(AUDIO_END)?;
        Self::token_counts(preprocessed)?
            .iter()
            .map(|&count| {
                if count == 0 {
                    return Err(ModelRegistryError::InvalidPreprocessedField {
                        field: "feature_token_counts".to_string(),
                    });
                }
                let mut tokens = Vec::with_capacity(count + 2);
                tokens.push(start);
                tokens.extend(std::iter::repeat_n(audio, count));
                tokens.push(end);
                Ok(
                    PromptReplacement::sequence(Modality::Audio, &placeholder, tokens)
                        .with_feature_span(1, count),
                )
            })
            .collect()
    }

    /// One replacement per image: `<|image_start|>`, one `<|patch|>` per
    /// merged token, `<|image_end|>`, the tokens between them as the feature
    /// span.
    fn image_replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let placeholder = Self::token_for(metadata, "image_token_id")?;
        let patch = Self::config_token_id(metadata, "image_token_id")?;
        let start = metadata.token_id(IMAGE_START)?;
        let end = metadata.token_id(IMAGE_END)?;
        Self::token_counts(preprocessed)?
            .iter()
            .map(|&count| {
                if count == 0 {
                    return Err(ModelRegistryError::InvalidPreprocessedField {
                        field: "feature_token_counts".to_string(),
                    });
                }
                let mut tokens = Vec::with_capacity(count + 2);
                tokens.push(start);
                tokens.extend(std::iter::repeat_n(patch, count));
                tokens.push(end);
                Ok(
                    PromptReplacement::sequence(Modality::Image, &placeholder, tokens)
                        .with_feature_span(1, count),
                )
            })
            .collect()
    }

    /// The `[t, h, w]` row of item `item` in `video_grid_thw`.
    fn video_grid(
        preprocessed: &PreprocessedEncoderInputs,
        item: usize,
    ) -> RegistryResult<[usize; 3]> {
        let invalid = || ModelRegistryError::InvalidPreprocessedField {
            field: "video_grid_thw".to_string(),
        };
        match preprocessed.model_specific.get("video_grid_thw") {
            Some(ModelSpecificValue::IntTensor { data, shape })
                if shape.len() == 2 && shape[1] == 3 && data.len() == shape[0] * 3 =>
            {
                let row = data.get(item * 3..item * 3 + 3).ok_or_else(invalid)?;
                let dim = |value: i64| usize::try_from(value).map_err(|_| invalid());
                Ok([dim(row[0])?, dim(row[1])?, dim(row[2])?])
            }
            _ => Err(invalid()),
        }
    }

    /// The seconds stamped on each of `groups` temporal groups, as the
    /// reference computes them from the sampled frames: the time of the
    /// group's first frame (`source index / source fps`, the decoder's
    /// record), the last stamp repeated for a group the sampled frames did
    /// not fill. Without the decoder's record the frames are taken as evenly
    /// spaced at `sample_fps`.
    fn group_timestamps(
        sampling: Option<&VideoSamplingInfo>,
        groups: usize,
        temporal_patch_size: usize,
        sample_fps: f64,
    ) -> Vec<f64> {
        let step = temporal_patch_size.max(1);
        let mut stamps: Vec<f64> = match sampling.filter(|sampling| {
            sampling.source_fps.is_finite()
                && sampling.source_fps > 0.0
                && !sampling.frame_indices.is_empty()
        }) {
            Some(sampling) => sampling
                .frame_indices
                .iter()
                .step_by(step)
                .take(groups)
                .map(|&index| index as f64 / sampling.source_fps)
                .collect(),
            None => (0..groups)
                .map(|group| (group * step) as f64 / sample_fps)
                .collect(),
        };
        while stamps.len() < groups {
            stamps.push(stamps.last().copied().unwrap_or(0.0));
        }
        stamps
    }

    /// `seconds` written as the Python format string `format` writes it:
    /// `{timestamp:.Nf}` (half to even on the exact binary value, as the
    /// interpreter rounds) or a bare `{timestamp}` (the float's shortest
    /// repr), with the literal text around it.
    fn format_timestamp(format: &str, seconds: f64) -> RegistryResult<String> {
        let invalid = |reason: &str| ModelRegistryError::InvalidConfigField {
            field: "frame_timestamp_format".to_string(),
            reason: reason.to_string(),
        };
        let (prefix, rest) = format
            .split_once("{timestamp")
            .ok_or_else(|| invalid("no {timestamp} field"))?;
        let (spec, suffix) = rest
            .split_once('}')
            .ok_or_else(|| invalid("unterminated {timestamp} field"))?;
        if prefix.contains('{') || suffix.contains('{') {
            return Err(invalid("only one {timestamp} field is supported"));
        }
        let rendered = if spec.is_empty() {
            format!("{seconds:?}")
        } else {
            let decimals = spec
                .strip_prefix(":.")
                .and_then(|spec| spec.strip_suffix('f'))
                .and_then(|digits| digits.parse::<usize>().ok())
                .ok_or_else(|| invalid("only {timestamp:.Nf} and {timestamp} are supported"))?;
            format!("{seconds:.decimals$}")
        };
        Ok(format!("{prefix}{rendered}{suffix}"))
    }

    /// One replacement per clip: `<|vid_start|>`, then per temporal group its
    /// stamp and one `<|video|>` per merged token, `<|vid_frame_separator|>`
    /// between groups and `<|vid_end|>` after the last; the token runs are
    /// the feature ranges. Stamps come from the clip's sampled frames
    /// (`media`) and the video config's format, grouping and rate.
    fn video_replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        media: &[MediaItemInfo],
        config: &PreProcessorConfig,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let placeholder = Self::token_for(metadata, "video_token_id")?;
        let video = Self::config_token_id(metadata, "video_token_id")?;
        let start = metadata.token_id(VIDEO_START)?;
        let end = metadata.token_id(VIDEO_END)?;
        let separator = metadata.token_id(VIDEO_SEPARATOR)?;
        let merge = config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE).max(1);
        let temporal = config
            .temporal_patch_size
            .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE)
            .max(1);
        let sample_fps = config
            .get_extra::<f64>("fps")
            .filter(|fps| fps.is_finite() && *fps > 0.0)
            .unwrap_or_else(|| f64::from(DEFAULT_VIDEO_FPS));
        let format = config
            .get_extra::<String>("frame_timestamp_format")
            .unwrap_or_else(|| DEFAULT_FRAME_TIMESTAMP_FORMAT.to_string());
        Self::token_counts(preprocessed)?
            .iter()
            .enumerate()
            .map(|(item, &total)| {
                let [groups, grid_h, grid_w] = Self::video_grid(preprocessed, item)?;
                let per_group = grid_h * grid_w / (merge * merge);
                if groups == 0 || per_group == 0 || total != groups * per_group {
                    return Err(ModelRegistryError::InvalidPreprocessedField {
                        field: "video_grid_thw".to_string(),
                    });
                }
                let stamps = Self::group_timestamps(
                    media
                        .get(item)
                        .and_then(|info| info.video_sampling.as_ref()),
                    groups,
                    temporal,
                    sample_fps,
                );
                let mut tokens = vec![start];
                let mut ranges = Vec::with_capacity(groups);
                for (group, &seconds) in stamps.iter().enumerate() {
                    let stamp = Self::format_timestamp(&format, seconds)?;
                    tokens.extend(metadata.encode_plain("muse_spark", &stamp)?);
                    ranges.push(PlaceholderRange {
                        offset: tokens.len(),
                        length: per_group,
                    });
                    tokens.extend(std::iter::repeat_n(video, per_group));
                    tokens.push(if group + 1 < groups { separator } else { end });
                }
                Ok(
                    PromptReplacement::sequence(Modality::Video, &placeholder, tokens)
                        .with_feature_ranges(ranges),
                )
            })
            .collect()
    }
}

impl ModelProcessorSpec for MuseSparkSpec {
    fn name(&self) -> &'static str {
        "muse_spark"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        if metadata
            .config_model_type()
            .is_some_and(|model_type| model_type == MODEL_TYPE)
        {
            return true;
        }
        if metadata
            .config
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|architectures| {
                architectures
                    .iter()
                    .any(|arch| arch.as_str() == Some(ARCHITECTURE))
            })
        {
            return true;
        }
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("muse-spark") || id.contains("muse_spark")
    }

    /// The template renders parts in request order.
    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    /// Patch rows with per-item grids, which the engine reads item by item.
    fn encoder_input_format(&self) -> EncoderInputFormat {
        EncoderInputFormat::Packed
    }

    /// Images and frames travel as `[patches, features]` rows flat over the
    /// batch, sliced per item by `patches_per_image` / `patches_per_video`,
    /// with one `[t, h, w]` grid row per item, kept on the CPU; audio as the
    /// clip's padded samples, `[samples, 1]` rows flat over the batch and
    /// sliced per clip by `samples_per_audio`.
    fn hooks(&self, modality: Modality) -> Option<ModalityHooks> {
        match modality {
            Modality::Image => Some(ModalityHooks {
                placeholder: |metadata| Self::token_for(metadata, "image_token_id"),
                placeholder_id: |metadata| Self::config_token_id(metadata, "image_token_id"),
                replacements: |metadata, preprocessed, _, _| {
                    Self::image_replacements(metadata, preprocessed)
                },
                layouts: || {
                    EncoderFieldLayouts::new(
                        FieldLayout::flat("patches_per_image"),
                        HashMap::from([
                            ("image_grid_thw".to_string(), FieldLayout::Batched),
                            ("patches_per_image".to_string(), FieldLayout::Batched),
                        ]),
                    )
                },
                keep_on_cpu: &["image_grid_thw"],
                encoder_input_key: None,
                worker_expandable: false,
            }),
            Modality::Video => Some(ModalityHooks {
                placeholder: |metadata| Self::token_for(metadata, "video_token_id"),
                placeholder_id: |metadata| Self::config_token_id(metadata, "video_token_id"),
                replacements: Self::video_replacements,
                layouts: || {
                    EncoderFieldLayouts::new(
                        FieldLayout::flat("patches_per_video"),
                        HashMap::from([
                            ("video_grid_thw".to_string(), FieldLayout::Batched),
                            ("patches_per_video".to_string(), FieldLayout::Batched),
                        ]),
                    )
                },
                keep_on_cpu: &["video_grid_thw"],
                encoder_input_key: None,
                worker_expandable: false,
            }),
            Modality::Audio => Some(ModalityHooks {
                placeholder: |metadata| Self::token_for(metadata, "audio_token_id"),
                placeholder_id: |metadata| Self::config_token_id(metadata, "audio_token_id"),
                replacements: |metadata, preprocessed, _, _| {
                    Self::audio_replacements(metadata, preprocessed)
                },
                layouts: || {
                    EncoderFieldLayouts::new(
                        FieldLayout::flat(MUSE_SPARK_SAMPLES_PER_AUDIO_KEY),
                        HashMap::from([(
                            MUSE_SPARK_SAMPLES_PER_AUDIO_KEY.to_string(),
                            FieldLayout::Batched,
                        )]),
                    )
                },
                keep_on_cpu: &[],
                encoder_input_key: None,
                worker_expandable: false,
            }),
            Modality::ImageEmbeds => None,
        }
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        self.placeholder_token_for(metadata, Modality::Image)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        self.placeholder_token_id_for(metadata, Modality::Image)
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        let mut limits = HashMap::new();
        if metadata.config_u32(&["image_token_id"]).is_some() {
            limits.insert(Modality::Image, MAX_IMAGES_PER_REQUEST);
        }
        if metadata.config_u32(&["video_token_id"]).is_some() {
            limits.insert(Modality::Video, MAX_VIDEOS_PER_REQUEST);
        }
        if Self::has_audio_tower(metadata.config) {
            limits.insert(Modality::Audio, MAX_AUDIOS_PER_REQUEST);
        }
        Ok(limits)
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    /// No audio tower is a model without audio; a tower whose geometry the
    /// processor cannot serve is an error naming the field.
    fn audio_processor(
        &self,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Option<Box<dyn AudioPreProcessor>>> {
        if !Self::has_audio_tower(model_config) {
            return Ok(None);
        }
        let processor = MuseSparkAudioProcessor::from_configs(model_config, preprocessor_config)
            .map_err(Self::audio_config_error)?;
        Ok(Some(Box::new(processor)))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.prompt_replacements_for(metadata, preprocessed, Modality::Image)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match self.hooks(modality) {
            Some(hooks) => {
                (hooks.replacements)(metadata, preprocessed, &[], &PreProcessorConfig::default())
            }
            None => Err(Self::unsupported(modality)),
        }
    }

    /// The reference video processor's `fps`.
    fn default_video_sample_fps(&self) -> Option<f32> {
        Some(DEFAULT_VIDEO_FPS)
    }

    fn default_video_sample_fps_with(&self, config: Option<&PreProcessorConfig>) -> Option<f32> {
        Some(
            config
                .and_then(|config| config.get_extra::<f32>("fps"))
                .filter(|fps| fps.is_finite() && *fps > 0.0)
                .unwrap_or(DEFAULT_VIDEO_FPS),
        )
    }

    /// The reference video processor's `sample_frames`: `trunc(duration x
    /// fps)` frames, at most `num_frames`, in temporal groups, at torch's
    /// linspace positions.
    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::TemporalGroups {
            frame_factor: DEFAULT_TEMPORAL_PATCH_SIZE,
            max_frames: DEFAULT_VIDEO_FRAMES,
        }
    }

    fn video_frame_sampling_with(&self, config: Option<&PreProcessorConfig>) -> FrameSampling {
        FrameSampling::TemporalGroups {
            frame_factor: config
                .and_then(|config| config.temporal_patch_size)
                .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE)
                .max(1),
            max_frames: config
                .and_then(|config| config.get_extra::<usize>("num_frames"))
                .filter(|frames| *frames > 0)
                .unwrap_or(DEFAULT_VIDEO_FRAMES),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::{test_helpers::*, ModelRegistry},
        types::ImageSize,
    };

    const PATCH_ID: u32 = 200_092;
    const VIDEO_ID: u32 = 200_091;
    const IMAGE_START_ID: u32 = 200_080;
    const IMAGE_END_ID: u32 = 200_081;
    const VIDEO_START_ID: u32 = 200_082;
    const VIDEO_END_ID: u32 = 200_083;
    const VIDEO_SEPARATOR_ID: u32 = 200_087;
    const AUDIO_ID: u32 = 200_038;
    const AUDIO_START_ID: u32 = 201_463;
    const AUDIO_END_ID: u32 = 201_464;
    /// Where the byte-encoding test tokenizer puts plain text.
    const TEXT_BASE: u32 = 1000;

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            ("<|patch|>", PATCH_ID),
            ("<|video|>", VIDEO_ID),
            ("<|audio|>", AUDIO_ID),
            (IMAGE_START, IMAGE_START_ID),
            (IMAGE_END, IMAGE_END_ID),
            (VIDEO_START, VIDEO_START_ID),
            (VIDEO_END, VIDEO_END_ID),
            (VIDEO_SEPARATOR, VIDEO_SEPARATOR_ID),
            (AUDIO_START, AUDIO_START_ID),
            (AUDIO_END, AUDIO_END_ID),
        ])
        .with_byte_encoder(TEXT_BASE)
    }

    fn config() -> Value {
        json!({
            "model_type": "muse_spark",
            "architectures": ["MuseSparkForConditionalGeneration"],
            "image_token_id": PATCH_ID,
            "video_token_id": VIDEO_ID,
            "audio_token_id": AUDIO_ID,
            "audio_config": {"model_type": "muse_spark_audio", "patch_sizes": [16, 6, 5, 4]},
        })
    }

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "/models/local-checkpoint",
            tokenizer,
            config,
        }
    }

    fn stamp(text: &str) -> Vec<TokenId> {
        text.bytes()
            .map(|byte| (TEXT_BASE + u32::from(byte)) as TokenId)
            .collect()
    }

    /// The stamps of a replacement, decoded from the byte-encoding tokenizer.
    fn stamps(replacement: &PromptReplacement) -> String {
        let text: Vec<u8> = replacement
            .tokens
            .iter()
            .filter_map(|&token| u8::try_from(token - TEXT_BASE as TokenId).ok())
            .collect();
        String::from_utf8(text).unwrap()
    }

    /// The released checkpoints' `video_processor` section, with the clip's
    /// sampling rate the gateway adds.
    fn video_config() -> PreProcessorConfig {
        PreProcessorConfig::from_json(
            r#"{"patch_size": 14, "merge_size": 2, "temporal_patch_size": 2, "fps": 2.0,
                "num_frames": 600, "max_video_frame_tokens": 256,
                "frame_timestamp_format": "Time: {timestamp:.2f}s"}"#,
        )
        .unwrap()
    }

    fn video(groups: usize, tokens: usize) -> PreprocessedEncoderInputs {
        test_preprocessed_with_tokens(&[ImageSize::new(56, 56)], &[tokens]).with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![groups as i64, 4, 4], 1, 3),
        )
    }

    fn sampled(source_fps: f64, frame_indices: Vec<usize>) -> [MediaItemInfo; 1] {
        [MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps,
                frame_indices,
            }),
        }]
    }

    #[test]
    fn the_registry_resolves_the_family_by_model_type_architecture_or_id() {
        let tokenizer = tokenizer();
        let registry = ModelRegistry::new();
        let by_type = config();
        assert_eq!(
            registry
                .lookup(&metadata(&tokenizer, &by_type))
                .unwrap()
                .name(),
            "muse_spark"
        );
        let by_architecture = json!({"architectures": ["MuseSparkForConditionalGeneration"]});
        assert_eq!(
            registry
                .lookup(&metadata(&tokenizer, &by_architecture))
                .unwrap()
                .name(),
            "muse_spark"
        );
        let by_id = ModelMetadata {
            model_id: "meta-models/Muse-Spark-open-816B-A42B-NVFP4",
            tokenizer: &tokenizer,
            config: &json!({}),
        };
        assert_eq!(registry.lookup(&by_id).unwrap().name(), "muse_spark");
        let other = ModelMetadata {
            model_id: "some/other-model",
            tokenizer: &tokenizer,
            config: &json!({"model_type": "llama"}),
        };
        assert!(!MuseSparkSpec.matches(&other));
    }

    #[test]
    fn spec_facts() {
        let spec = MuseSparkSpec;
        assert_eq!(spec.media_part_order(), MediaPartOrder::Authored);
        assert_eq!(spec.encoder_input_format(), EncoderInputFormat::Packed);
        for modality in [Modality::Image, Modality::Video] {
            assert!(!spec.worker_expandable(modality));
            assert_eq!(spec.rendered_marker_for(modality), None);
        }
        assert_eq!(spec.default_video_sample_fps(), Some(2.0));
        assert_eq!(
            spec.video_frame_sampling(),
            FrameSampling::TemporalGroups {
                frame_factor: 2,
                max_frames: 96
            }
        );
        let config = video_config();
        assert_eq!(
            spec.video_frame_sampling_with(Some(&config)),
            FrameSampling::TemporalGroups {
                frame_factor: 2,
                max_frames: 600
            }
        );
        assert_eq!(spec.default_video_sample_fps_with(Some(&config)), Some(2.0));
        assert_eq!(spec.default_video_sample_fps_with(None), Some(2.0));
        let image = spec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(image.encoder_input, FieldLayout::flat("patches_per_image"));
        assert_eq!(
            image.model_specific.get("image_grid_thw"),
            Some(&FieldLayout::Batched)
        );
        let video = spec.encoder_field_layouts_for(Modality::Video);
        assert_eq!(video.encoder_input, FieldLayout::flat("patches_per_video"));
        assert_eq!(
            video.model_specific.get("video_grid_thw"),
            Some(&FieldLayout::Batched)
        );
        assert!(!video.model_specific.contains_key("patches_per_image"));
        assert_eq!(
            spec.keep_on_cpu_keys_for(Modality::Image),
            ["image_grid_thw"]
        );
        assert_eq!(
            spec.keep_on_cpu_keys_for(Modality::Video),
            ["video_grid_thw"]
        );
        assert_eq!(spec.encoder_input_key_for(Modality::Image), None);
    }

    #[test]
    fn limits_follow_the_config_tokens() {
        let tokenizer = tokenizer();
        let config = config();
        let limits = MuseSparkSpec
            .modality_limits(&metadata(&tokenizer, &config))
            .unwrap();
        assert_eq!(limits.get(&Modality::Image), Some(&MAX_IMAGES_PER_REQUEST));
        assert_eq!(limits.get(&Modality::Video), Some(&MAX_VIDEOS_PER_REQUEST));
        let text_and_image = json!({"model_type": "muse_spark", "image_token_id": PATCH_ID});
        let limits = MuseSparkSpec
            .modality_limits(&metadata(&tokenizer, &text_and_image))
            .unwrap();
        assert!(!limits.contains_key(&Modality::Video));
    }

    #[test]
    fn placeholders_are_the_anchor_tokens_the_template_writes() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        let spec = MuseSparkSpec;
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Image)
                .unwrap(),
            "<|patch|>"
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Image)
                .unwrap(),
            PATCH_ID as TokenId
        );
        assert_eq!(spec.placeholder_token(&metadata).unwrap(), "<|patch|>");
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Video)
                .unwrap(),
            "<|video|>"
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Video)
                .unwrap(),
            VIDEO_ID as TokenId
        );
        assert!(matches!(
            spec.placeholder_token_for(&metadata, Modality::ImageEmbeds),
            Err(ModelRegistryError::UnsupportedModality { .. })
        ));
    }

    #[test]
    fn image_replacement_wraps_the_patches_with_an_explicit_feature_span() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        let replacements = MuseSparkSpec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(28, 28); 2], &[3, 1]),
                Modality::Image,
            )
            .unwrap();
        assert_eq!(replacements.len(), 2);
        let patch = PATCH_ID as TokenId;
        assert_eq!(
            replacements[0].tokens,
            vec![
                IMAGE_START_ID as TokenId,
                patch,
                patch,
                patch,
                IMAGE_END_ID as TokenId
            ]
        );
        assert_eq!(
            replacements[0].feature_ranges,
            Some(vec![PlaceholderRange {
                offset: 1,
                length: 3
            }])
        );
        assert_eq!(replacements[0].placeholder_token, "<|patch|>");
        assert_eq!(replacements[0].structural_prefix, 0);
        assert_eq!(replacements[1].tokens.len(), 3);
        assert!(matches!(
            MuseSparkSpec.prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(28, 28)], &[0]),
            ),
            Err(ModelRegistryError::InvalidPreprocessedField { .. })
        ));
    }

    /// Two temporal groups of 4 tokens from frames 0, 15, 30, 45 of a 30 fps
    /// clip: stamps at 0.00 s and 1.00 s, the separator between the groups,
    /// the end marker after the last, the token runs as the feature ranges.
    #[test]
    fn video_replacement_stamps_each_group_from_the_sampled_frames() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        let replacements = MuseSparkSpec
            .prompt_replacements_with_media(
                &metadata,
                &video(2, 8),
                Modality::Video,
                &sampled(30.0, vec![0, 15, 30, 45]),
                &video_config(),
            )
            .unwrap();
        let fill = VIDEO_ID as TokenId;
        let mut expected = vec![VIDEO_START_ID as TokenId];
        expected.extend(stamp("Time: 0.00s"));
        let first_offset = expected.len();
        expected.extend(std::iter::repeat_n(fill, 4));
        expected.push(VIDEO_SEPARATOR_ID as TokenId);
        expected.extend(stamp("Time: 1.00s"));
        let second_offset = expected.len();
        expected.extend(std::iter::repeat_n(fill, 4));
        expected.push(VIDEO_END_ID as TokenId);
        assert_eq!(replacements[0].tokens, expected);
        assert_eq!(
            replacements[0].feature_ranges,
            Some(vec![
                PlaceholderRange {
                    offset: first_offset,
                    length: 4
                },
                PlaceholderRange {
                    offset: second_offset,
                    length: 4
                },
            ])
        );
        assert_eq!(replacements[0].placeholder_token, "<|video|>");
    }

    /// Three sampled frames of a 24 fps clip: the second group holds frame 2
    /// alone (the processor repeats it) and is stamped with its time; a
    /// single sampled frame gives one stamped group closed by the end marker;
    /// without the decoder's record the stamps assume the sampling rate.
    #[test]
    fn video_stamps_pad_like_the_reference() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        let two_groups = MuseSparkSpec
            .prompt_replacements_with_media(
                &metadata,
                &video(2, 8),
                Modality::Video,
                &sampled(24.0, vec![0, 12, 23]),
                &video_config(),
            )
            .unwrap();
        assert_eq!(stamps(&two_groups[0]), "Time: 0.00sTime: 0.96s");

        let one_group = MuseSparkSpec
            .prompt_replacements_with_media(
                &metadata,
                &video(1, 4),
                Modality::Video,
                &sampled(24.0, vec![0]),
                &video_config(),
            )
            .unwrap();
        let mut expected = vec![VIDEO_START_ID as TokenId];
        expected.extend(stamp("Time: 0.00s"));
        expected.extend(std::iter::repeat_n(VIDEO_ID as TokenId, 4));
        expected.push(VIDEO_END_ID as TokenId);
        assert_eq!(one_group[0].tokens, expected);

        // More groups than sampled frames: the last stamp repeats.
        let padded = MuseSparkSpec
            .prompt_replacements_with_media(
                &metadata,
                &video(2, 8),
                Modality::Video,
                &sampled(24.0, vec![6]),
                &video_config(),
            )
            .unwrap();
        assert_eq!(stamps(&padded[0]), "Time: 0.25sTime: 0.25s");

        let assumed = MuseSparkSpec
            .prompt_replacements_with_media(
                &metadata,
                &video(2, 8),
                Modality::Video,
                &[MediaItemInfo::default()],
                &video_config(),
            )
            .unwrap();
        assert_eq!(stamps(&assumed[0]), "Time: 0.00sTime: 1.00s");

        // Without a video config the reference class's one-decimal stamp.
        let class_default = MuseSparkSpec
            .prompt_replacements_for(&metadata, &video(2, 8), Modality::Video)
            .unwrap();
        assert_eq!(stamps(&class_default[0]), "Time: 0.0sTime: 1.0s");
    }

    #[test]
    fn timestamps_render_like_python_format() {
        let fmt =
            |format: &str, seconds: f64| MuseSparkSpec::format_timestamp(format, seconds).unwrap();
        assert_eq!(fmt("Time: {timestamp:.2f}s", 0.0), "Time: 0.00s");
        assert_eq!(fmt("Time: {timestamp:.2f}s", 9.0 / 20.0), "Time: 0.45s");
        assert_eq!(fmt("Time: {timestamp:.1f}s", 9.0 / 20.0), "Time: 0.5s");
        assert_eq!(fmt("Time: {timestamp:.2f}s", 1234.5678), "Time: 1234.57s");
        // Exact binary ties round half to even, as Python's format does;
        // 2.675 sits below its midpoint in binary and rounds down.
        assert_eq!(fmt("{timestamp:.2f}", 0.125), "0.12");
        assert_eq!(fmt("{timestamp:.2f}", 0.375), "0.38");
        assert_eq!(fmt("{timestamp:.0f}", 2.5), "2");
        assert_eq!(fmt("{timestamp:.2f}", 2.675), "2.67");
        assert_eq!(fmt("<{timestamp}>", 2.5), "<2.5>");
        assert_eq!(fmt("<{timestamp}>", 1.0), "<1.0>");
        assert!(matches!(
            MuseSparkSpec::format_timestamp("Time: {ts}", 1.0),
            Err(ModelRegistryError::InvalidConfigField { .. })
        ));
        assert!(matches!(
            MuseSparkSpec::format_timestamp("{timestamp:>8}", 1.0),
            Err(ModelRegistryError::InvalidConfigField { .. })
        ));
        assert!(matches!(
            MuseSparkSpec::format_timestamp("{timestamp:.2f} {timestamp}", 1.0),
            Err(ModelRegistryError::InvalidConfigField { .. })
        ));
    }

    #[test]
    fn video_replacement_rejects_a_grid_that_does_not_match_the_token_count() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        assert!(matches!(
            MuseSparkSpec.prompt_replacements_for(&metadata, &video(2, 7), Modality::Video),
            Err(ModelRegistryError::InvalidPreprocessedField { .. })
        ));
        let no_grid = test_preprocessed_with_tokens(&[ImageSize::new(56, 56)], &[8]);
        assert!(matches!(
            MuseSparkSpec.prompt_replacements_for(&metadata, &no_grid, Modality::Video),
            Err(ModelRegistryError::InvalidPreprocessedField { .. })
        ));
        assert!(matches!(
            MuseSparkSpec.prompt_replacements_for(&metadata, &no_grid, Modality::ImageEmbeds),
            Err(ModelRegistryError::UnsupportedModality { .. })
        ));
    }

    #[test]
    fn audio_replacements_wrap_the_token_run_in_the_soft_delimiters() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        let spec = MuseSparkSpec;
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Audio)
                .unwrap(),
            "<|audio|>"
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Audio)
                .unwrap(),
            AUDIO_ID as TokenId
        );
        assert!(!spec.worker_expandable(Modality::Audio));
        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(
                    &[ImageSize::new(1, 1), ImageSize::new(1, 1)],
                    &[3, 1],
                ),
                Modality::Audio,
            )
            .unwrap();
        assert_eq!(replacements.len(), 2);
        assert_eq!(replacements[0].placeholder_token, "<|audio|>");
        assert_eq!(
            replacements[0].tokens,
            vec![
                AUDIO_START_ID as TokenId,
                AUDIO_ID as TokenId,
                AUDIO_ID as TokenId,
                AUDIO_ID as TokenId,
                AUDIO_END_ID as TokenId
            ]
        );
        assert_eq!(
            replacements[0].feature_ranges,
            Some(vec![PlaceholderRange {
                offset: 1,
                length: 3
            }])
        );
        assert_eq!(
            replacements[1].tokens,
            vec![
                AUDIO_START_ID as TokenId,
                AUDIO_ID as TokenId,
                AUDIO_END_ID as TokenId
            ]
        );
        let layouts = spec.encoder_field_layouts_for(Modality::Audio);
        assert_eq!(
            layouts.encoder_input,
            FieldLayout::flat(MUSE_SPARK_SAMPLES_PER_AUDIO_KEY)
        );
        assert_eq!(
            layouts.model_specific,
            HashMap::from([(
                MUSE_SPARK_SAMPLES_PER_AUDIO_KEY.to_string(),
                FieldLayout::Batched
            )])
        );
        assert!(spec.keep_on_cpu_keys_for(Modality::Audio).is_empty());
        let bare = TestTokenizer::new(&[("<|audio|>", AUDIO_ID)]);
        let metadata = self::metadata(&bare, &config);
        assert_eq!(
            spec.prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(1, 1)], &[1]),
                Modality::Audio
            )
            .unwrap_err(),
            ModelRegistryError::TokenNotFound {
                token: AUDIO_START.to_string()
            }
        );
    }

    #[test]
    fn audio_is_served_only_with_the_tower_and_its_processor_follows_the_configs() {
        use std::sync::Arc;

        use bytes::Bytes;

        use crate::{
            audio::DecodedAudio,
            types::{AudioClip, AudioSource},
        };

        let tokenizer = tokenizer();
        let config = config();
        let spec = MuseSparkSpec;
        let limits = spec
            .modality_limits(&metadata(&tokenizer, &config))
            .unwrap();
        assert_eq!(limits.get(&Modality::Audio), Some(&MAX_AUDIOS_PER_REQUEST));
        let without_tower = json!({
            "model_type": "muse_spark",
            "image_token_id": PATCH_ID,
            "audio_token_id": AUDIO_ID,
        });
        assert!(!spec
            .modality_limits(&metadata(&tokenizer, &without_tower))
            .unwrap()
            .contains_key(&Modality::Audio));

        let default = PreProcessorConfig::default();
        let processor = spec
            .audio_processor(&config, &default)
            .unwrap()
            .expect("a checkpoint with an audio tower has an audio processor");
        let clip = Arc::new(AudioClip::new(
            Bytes::from_static(b"audio"),
            DecodedAudio {
                samples: vec![0.5; 1_921],
                sample_rate: 24_000,
            },
            AudioSource::InlineBytes,
            "audio-hash".to_string(),
        ));
        let output = processor.preprocess(&[clip]).unwrap();
        assert_eq!(output.feature_token_counts, vec![2]);
        assert_eq!(output.encoder_input.shape(), &[2 * 1_920, 1]);
        assert!(spec
            .audio_processor(&without_tower, &default)
            .unwrap()
            .is_none());

        let extractor =
            PreProcessorConfig::from_json(r#"{"samples_per_token": 1600, "sampling_rate": 24000}"#)
                .unwrap();
        match spec.audio_processor(&config, &extractor) {
            Err(ModelRegistryError::InvalidConfigField { field, .. }) => {
                assert_eq!(field, "samples_per_token");
            }
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("a disagreeing samples_per_token must be refused"),
        }
    }
}
