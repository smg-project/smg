//! Gemma 4 prompt contract.
//!
//! Mirrors vLLM's Gemma 4 multimodal processor, which is what the engine
//! expects around the tensors [`crate::vision::Gemma4Processor`] produces.
//! The chat template renders one `<|image|>` per image and one `<|video|>`
//! per video. An image's anchor becomes `<|image>` (begin), one `<|image|>`
//! per soft token the pooled patch grid yields, and `<image|>` (end): 258
//! tokens for a square image. A video's anchor becomes one group per sampled
//! frame, `MM:SS <|image> <|video|> x n <image|>` with the frames' timestamps
//! (the source frame index over the source rate) and the groups separated by
//! spaces, the first timestamp encoded as `"MM:SS "` and the later ones as
//! `" MM:SS "`, as vLLM tokenizes them. The begin/end markers and timestamps
//! are structural; only the soft tokens are feature positions.
//!
//! Frames are sampled as the engine's default loader samples them: every
//! frame of a clip with at most 32, else 32 spread evenly across the clip.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    media::FrameSampling,
    registry::{
        MediaItemInfo, MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError,
        RegistryResult,
    },
    types::{
        EncoderFieldLayouts, FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId,
        VideoSamplingInfo,
    },
    vision::PreProcessorConfig,
};

/// The engine's default loader keeps at most this many frames of a clip.
const MAX_SAMPLED_FRAMES: usize = 32;
/// The rate vLLM assumes for timestamps when the decoder reported none.
const FALLBACK_FPS: f64 = 24.0;
/// vLLM leaves both counts unlimited for this family; these caps are the
/// per-request defaults a deployment raises or lowers with
/// `SMG_IMAGE_MAX_COUNT` / `SMG_VIDEO_MAX_COUNT` or the worker's item cap.
const MAX_IMAGES_PER_REQUEST: usize = 128;
const MAX_VIDEOS_PER_REQUEST: usize = 8;

pub(super) struct Gemma4Spec;

impl Gemma4Spec {
    fn config_token_id(metadata: &ModelMetadata, field: &str) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&[field])
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: field.to_string(),
            })
    }

    fn token_for(metadata: &ModelMetadata, field: &str) -> RegistryResult<String> {
        let id = Self::config_token_id(metadata, field)?;
        metadata
            .tokenizer
            .id_to_token(id as u32)
            .ok_or_else(|| ModelRegistryError::TokenNotFound {
                token: format!("{field}:{id}"),
            })
    }

    fn encode(metadata: &ModelMetadata, text: &str) -> RegistryResult<Vec<TokenId>> {
        let ids = metadata.tokenizer.encode_text(text).ok_or_else(|| {
            ModelRegistryError::TextEncodingFailed {
                spec: "gemma4",
                text: text.to_string(),
            }
        })?;
        Ok(ids.into_iter().map(|id| id as TokenId).collect())
    }

    fn unsupported(modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: "gemma4",
            modality,
        }
    }

    /// `[begin] + n x token + [end]`, the soft tokens as the feature span.
    fn wrapped(
        placeholder: &str,
        modality: Modality,
        begin: TokenId,
        token: TokenId,
        end: TokenId,
        count: usize,
    ) -> PromptReplacement {
        let mut tokens = Vec::with_capacity(count + 2);
        tokens.push(begin);
        tokens.extend(std::iter::repeat_n(token, count));
        tokens.push(end);
        PromptReplacement::sequence(modality, placeholder, tokens).with_feature_span(1, count)
    }

    /// Frames per clip, as the paired processor emits them.
    fn frame_counts(preprocessed: &PreprocessedEncoderInputs) -> RegistryResult<Vec<usize>> {
        let invalid = || ModelRegistryError::InvalidPreprocessedField {
            field: "video_frame_counts".to_string(),
        };
        let counts = match preprocessed.model_specific.get("video_frame_counts") {
            Some(ModelSpecificValue::IntTensor { data, .. }) => data.clone(),
            Some(ModelSpecificValue::IntVec(data)) => data.clone(),
            _ => return Err(invalid()),
        };
        if counts.len() != preprocessed.feature_token_counts.len() {
            return Err(invalid());
        }
        counts
            .into_iter()
            .map(|count| {
                usize::try_from(count)
                    .ok()
                    .filter(|count| *count > 0)
                    .ok_or_else(invalid)
            })
            .collect()
    }

    /// `MM:SS` of a frame's time, truncated like the reference's `int()`.
    fn timestamp(seconds: f64) -> String {
        let minutes = (seconds / 60.0).floor() as u64;
        let seconds = seconds.rem_euclid(60.0).floor() as u64;
        format!("{minutes:02}:{seconds:02}")
    }

    /// The seconds of each sampled frame: its source index over the source
    /// rate, or the frame's ordinal at the fallback rate when the decoder
    /// reported no sampling, as vLLM falls back.
    fn frame_seconds(frames: usize, sampling: Option<&VideoSamplingInfo>) -> Vec<f64> {
        (0..frames)
            .map(|frame| match sampling {
                Some(sampling) => {
                    sampling.frame_indices.get(frame).copied().unwrap_or(frame) as f64
                        / sampling.source_fps
                }
                None => frame as f64 / FALLBACK_FPS,
            })
            .collect()
    }

    fn video_replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        sampling: &[Option<&VideoSamplingInfo>],
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let placeholder = Self::token_for(metadata, "video_token_id")?;
        let video = Self::config_token_id(metadata, "video_token_id")?;
        let begin = Self::config_token_id(metadata, "boi_token_id")?;
        let end = Self::config_token_id(metadata, "eoi_token_id")?;
        let frame_counts = Self::frame_counts(preprocessed)?;
        preprocessed
            .feature_token_counts
            .iter()
            .zip(frame_counts)
            .enumerate()
            .map(|(clip, (&total, frames))| {
                if !total.is_multiple_of(frames) || total == 0 {
                    return Err(ModelRegistryError::InvalidPreprocessedField {
                        field: "video_frame_counts token layout".to_string(),
                    });
                }
                let per_frame = total / frames;
                let seconds = Self::frame_seconds(frames, sampling.get(clip).copied().flatten());
                let mut tokens = Vec::with_capacity(frames * (per_frame + 9));
                let mut ranges = Vec::with_capacity(frames);
                for (frame, seconds) in seconds.into_iter().enumerate() {
                    let stamp = Self::timestamp(seconds);
                    let prefix = if frame == 0 {
                        format!("{stamp} ")
                    } else {
                        format!(" {stamp} ")
                    };
                    tokens.extend(Self::encode(metadata, &prefix)?);
                    tokens.push(begin);
                    ranges.push(PlaceholderRange {
                        offset: tokens.len(),
                        length: per_frame,
                    });
                    tokens.extend(std::iter::repeat_n(video, per_frame));
                    tokens.push(end);
                }
                Ok(
                    PromptReplacement::sequence(Modality::Video, &placeholder, tokens)
                        .with_feature_ranges(ranges),
                )
            })
            .collect()
    }
}

impl ModelProcessorSpec for Gemma4Spec {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        metadata.config_model_type() == Some("gemma4")
            || id.contains("gemma-4")
            || id.contains("gemma4")
    }

    /// The template renders the parts in the order they were written.
    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    /// `<|image|>` and `<|video|>` are the single tokens vLLM's prompt
    /// updates target.
    fn worker_expandable(&self, modality: Modality) -> bool {
        matches!(modality, Modality::Image | Modality::Video)
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        Self::token_for(metadata, "image_token_id")
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::config_token_id(metadata, "image_token_id")
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            Modality::Video => Self::token_for(metadata, "video_token_id"),
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image => self.placeholder_token_id(metadata),
            Modality::Video => Self::config_token_id(metadata, "video_token_id"),
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        // Images need their token ids in the config; video additionally its
        // own token, which a text-and-image derivative may lack.
        let mut limits = HashMap::new();
        if metadata.config_u32(&["image_token_id"]).is_some()
            && metadata.config_u32(&["boi_token_id"]).is_some()
            && metadata.config_u32(&["eoi_token_id"]).is_some()
        {
            limits.insert(Modality::Image, MAX_IMAGES_PER_REQUEST);
            if metadata.config_u32(&["video_token_id"]).is_some() {
                limits.insert(Modality::Video, MAX_VIDEOS_PER_REQUEST);
            }
        }
        Ok(limits)
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let placeholder = self.placeholder_token(metadata)?;
        let image = Self::config_token_id(metadata, "image_token_id")?;
        let begin = Self::config_token_id(metadata, "boi_token_id")?;
        let end = Self::config_token_id(metadata, "eoi_token_id")?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| Self::wrapped(&placeholder, Modality::Image, begin, image, end, count))
            .collect())
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.prompt_replacements(metadata, preprocessed),
            Modality::Video => Self::video_replacements(metadata, preprocessed, &[]),
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn prompt_replacements_with_media(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
        media: &[MediaItemInfo],
        _preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Video => {
                let sampling: Vec<Option<&VideoSamplingInfo>> = media
                    .iter()
                    .map(|item| {
                        item.video_sampling.as_ref().filter(|sampling| {
                            sampling.source_fps.is_finite() && sampling.source_fps > 0.0
                        })
                    })
                    .collect();
                Self::video_replacements(metadata, preprocessed, &sampling)
            }
            _ => self.prompt_replacements_for(metadata, preprocessed, modality),
        }
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // vLLM's `_get_mm_fields_config`: images batched per item, video
        // frames flat over the batch and sliced per clip by its frame count.
        HashMap::from([
            ("pixel_values".to_string(), FieldLayout::Batched),
            ("pixel_position_ids".to_string(), FieldLayout::Batched),
            (
                "pixel_values_videos".to_string(),
                FieldLayout::flat("video_frame_counts"),
            ),
            (
                "pixel_position_ids_videos".to_string(),
                FieldLayout::flat("video_frame_counts"),
            ),
            ("video_frame_counts".to_string(), FieldLayout::Batched),
        ])
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Video => EncoderFieldLayouts::new(
                FieldLayout::flat("video_frame_counts"),
                HashMap::from([
                    (
                        "pixel_position_ids_videos".to_string(),
                        FieldLayout::flat("video_frame_counts"),
                    ),
                    ("video_frame_counts".to_string(), FieldLayout::Batched),
                ]),
            ),
            _ => EncoderFieldLayouts::new(
                FieldLayout::Batched,
                HashMap::from([("pixel_position_ids".to_string(), FieldLayout::Batched)]),
            ),
        }
    }

    /// The model's forward takes frames as `pixel_values_videos`.
    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        match modality {
            Modality::Video => Some("pixel_values_videos".to_string()),
            _ => None,
        }
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["video_frame_counts".to_string()]
    }

    fn keep_on_cpu_keys_for(&self, modality: Modality) -> Vec<String> {
        match modality {
            Modality::Video => self.keep_on_cpu_keys(),
            _ => Vec::new(),
        }
    }

    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::UpTo {
            max_frames: MAX_SAMPLED_FRAMES,
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

    const IMAGE: &str = "<|image|>";
    const VIDEO: &str = "<|video|>";
    const BEGIN: &str = "<|image>";
    const END: &str = "<image|>";
    const IMAGE_ID: u32 = 258880;
    const VIDEO_ID: u32 = 258884;
    const BEGIN_ID: u32 = 255999;
    const END_ID: u32 = 258882;
    /// Plain text encodes as one id per byte above this base.
    const TEXT_BASE: u32 = 1000;

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            (IMAGE, IMAGE_ID),
            (VIDEO, VIDEO_ID),
            (BEGIN, BEGIN_ID),
            (END, END_ID),
        ])
        .with_byte_encoder(TEXT_BASE)
    }

    fn config() -> Value {
        json!({
            "model_type": "gemma4",
            "image_token_id": IMAGE_ID,
            "boi_token_id": BEGIN_ID,
            "eoi_token_id": END_ID,
            "video_token_id": VIDEO_ID,
            "vision_config": {"patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 280}
        })
    }

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "/models/local-checkpoint",
            tokenizer,
            config,
        }
    }

    fn text(s: &str) -> Vec<TokenId> {
        s.bytes()
            .map(|b| (TEXT_BASE + u32::from(b)) as TokenId)
            .collect()
    }

    fn video_input(frames: usize, per_frame: usize) -> PreprocessedEncoderInputs {
        test_preprocessed_with_tokens(&[ImageSize::new(320, 240)], &[frames * per_frame])
            .with_extra(
                "video_frame_counts",
                ModelSpecificValue::int_1d(vec![frames as i64]),
            )
    }

    #[test]
    fn matches_the_family_by_model_type_and_id() {
        let tokenizer = tokenizer();
        let config = config();
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata(&tokenizer, &config))
            .expect("gemma4 by model_type");
        assert_eq!(spec.name(), "gemma4");
        let neutral = json!({});
        for model_id in ["google/gemma-4-26B-A4B-it", "GEMMA4-local"] {
            let by_id = ModelMetadata {
                model_id,
                tokenizer: &tokenizer,
                config: &neutral,
            };
            assert!(Gemma4Spec.matches(&by_id), "{model_id}");
        }
        let gemma3 = json!({"model_type": "gemma3"});
        assert!(!Gemma4Spec.matches(&ModelMetadata {
            model_id: "google/gemma-3-4b-it",
            tokenizer: &tokenizer,
            config: &gemma3,
        }));
    }

    #[test]
    fn image_anchor_expands_to_begin_soft_tokens_end() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        let spec = Gemma4Spec;
        assert_eq!(spec.placeholder_token(&metadata).unwrap(), IMAGE);
        assert_eq!(
            spec.placeholder_token_id(&metadata).unwrap(),
            IMAGE_ID as TokenId
        );
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Video)
                .unwrap(),
            VIDEO
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Video)
                .unwrap(),
            VIDEO_ID as TokenId
        );
        assert!(spec
            .placeholder_token_for(&metadata, Modality::Audio)
            .is_err());

        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(
                    &[ImageSize::new(512, 512), ImageSize::new(1920, 1080)],
                    &[256, 264],
                ),
            )
            .unwrap();
        assert_eq!(replacements.len(), 2);
        for (replacement, count) in replacements.iter().zip([256usize, 264]) {
            assert_eq!(replacement.modality, Modality::Image);
            assert_eq!(replacement.placeholder_token, IMAGE);
            assert_eq!(replacement.tokens.len(), count + 2);
            assert_eq!(replacement.tokens[0], BEGIN_ID as TokenId);
            assert!(replacement.tokens[1..=count]
                .iter()
                .all(|&t| t == IMAGE_ID as TokenId));
            assert_eq!(replacement.tokens[count + 1], END_ID as TokenId);
            assert_eq!(
                replacement.feature_ranges,
                Some(vec![PlaceholderRange {
                    offset: 1,
                    length: count
                }])
            );
            assert_eq!(replacement.structural_prefix, 0);
        }
    }

    #[test]
    fn video_anchor_expands_to_timestamped_frame_groups() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        // 16 frames of a 2 s clip at 8 fps, 63 soft tokens each.
        let media = vec![MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps: 8.0,
                frame_indices: (0..16).collect(),
            }),
        }];
        let replacement = Gemma4Spec
            .prompt_replacements_with_media(
                &metadata,
                &video_input(16, 63),
                Modality::Video,
                &media,
                &PreProcessorConfig::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(replacement.modality, Modality::Video);
        assert_eq!(replacement.placeholder_token, VIDEO);
        let tokens = &replacement.tokens;
        let ranges = replacement.feature_ranges.as_ref().unwrap();
        assert_eq!(ranges.len(), 16);
        // First group: "00:00 " then begin, 63 soft tokens, end.
        let first_stamp = text("00:00 ");
        assert_eq!(&tokens[..first_stamp.len()], first_stamp.as_slice());
        assert_eq!(tokens[first_stamp.len()], BEGIN_ID as TokenId);
        assert_eq!(ranges[0].offset, first_stamp.len() + 1);
        assert_eq!(ranges[0].length, 63);
        let first_end = ranges[0].offset + 63;
        assert_eq!(tokens[first_end], END_ID as TokenId);
        // Second group starts with " 00:00 " (frame 1 at 0.125 s).
        let second_stamp = text(" 00:00 ");
        assert_eq!(
            &tokens[first_end + 1..first_end + 1 + second_stamp.len()],
            second_stamp.as_slice()
        );
        // Frame 8 is at 1.0 s.
        let eighth = ranges[8].offset - 1 - text(" 00:01 ").len();
        assert_eq!(
            &tokens[eighth..ranges[8].offset - 1],
            text(" 00:01 ").as_slice()
        );
        assert_eq!(
            tokens.len(),
            first_stamp.len() + 15 * second_stamp.len() + 16 * (63 + 2)
        );
        assert!(ranges
            .iter()
            .all(|range| tokens[range.offset..range.offset + range.length]
                .iter()
                .all(|&t| t == VIDEO_ID as TokenId)));
    }

    #[test]
    fn video_without_sampling_counts_frames_at_the_fallback_rate() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        // 32 frames at the 24 fps fallback: frame 24 is at 00:01, frame 31 at 00:01.
        let replacement = Gemma4Spec
            .prompt_replacements_for(&metadata, &video_input(32, 66), Modality::Video)
            .unwrap()
            .pop()
            .unwrap();
        let ranges = replacement.feature_ranges.unwrap();
        assert_eq!(ranges.len(), 32);
        let stamp = |frame: usize| {
            let end = ranges[frame].offset - 1;
            replacement.tokens[end - text(" 00:00 ").len()..end].to_vec()
        };
        assert_eq!(stamp(23), text(" 00:00 "));
        assert_eq!(stamp(24), text(" 00:01 "));
        assert_eq!(stamp(31), text(" 00:01 "));
        // Sampling with the clip's own indices: frame 2 of [0, 30, 60] at 30 fps is 00:02.
        let media = vec![MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps: 30.0,
                frame_indices: vec![0, 30, 60],
            }),
        }];
        let sampled = Gemma4Spec
            .prompt_replacements_with_media(
                &metadata,
                &video_input(3, 70),
                Modality::Video,
                &media,
                &PreProcessorConfig::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let ranges = sampled.feature_ranges.unwrap();
        let end = ranges[2].offset - 1;
        assert_eq!(
            sampled.tokens[end - text(" 00:02 ").len()..end],
            text(" 00:02 ")
        );
        assert_eq!(Gemma4Spec::timestamp(59.999), "00:59");
        assert_eq!(Gemma4Spec::timestamp(125.0), "02:05");
    }

    #[test]
    fn video_layout_errors_fail_loudly() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        // No frame counts.
        let missing = test_preprocessed_with_tokens(&[ImageSize::new(320, 240)], &[126]);
        assert!(matches!(
            Gemma4Spec.prompt_replacements_for(&metadata, &missing, Modality::Video),
            Err(ModelRegistryError::InvalidPreprocessedField { .. })
        ));
        // Tokens not divisible by the frame count.
        assert!(matches!(
            Gemma4Spec.prompt_replacements_for(
                &metadata,
                &video_input(5, 63)
                    .with_extra("video_frame_counts", ModelSpecificValue::int_1d(vec![4])),
                Modality::Video
            ),
            Err(ModelRegistryError::InvalidPreprocessedField { .. })
        ));
        assert!(matches!(
            Gemma4Spec.prompt_replacements_for(&metadata, &video_input(2, 63), Modality::Audio),
            Err(ModelRegistryError::UnsupportedModality { .. })
        ));
        // Missing config ids.
        let bare = json!({"model_type": "gemma4"});
        let bare_metadata = metadata_for(&tokenizer, &bare);
        assert!(matches!(
            Gemma4Spec.placeholder_token_id(&bare_metadata),
            Err(ModelRegistryError::MissingConfigField { .. })
        ));
    }

    fn metadata_for<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        metadata(tokenizer, config)
    }

    #[test]
    fn limits_layouts_and_hooks() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = metadata(&tokenizer, &config);
        let spec = Gemma4Spec;
        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([(Modality::Image, 128), (Modality::Video, 8)])
        );
        let image_only = json!({"model_type": "gemma4", "image_token_id": IMAGE_ID, "boi_token_id": BEGIN_ID, "eoi_token_id": END_ID});
        assert_eq!(
            spec.modality_limits(&metadata_for(&tokenizer, &image_only))
                .unwrap(),
            HashMap::from([(Modality::Image, 128)])
        );
        let text_only = json!({"model_type": "gemma4"});
        assert!(spec
            .modality_limits(&metadata_for(&tokenizer, &text_only))
            .unwrap()
            .is_empty());

        assert_eq!(spec.media_part_order(), MediaPartOrder::Authored);
        assert!(spec.worker_expandable(Modality::Image));
        assert!(spec.worker_expandable(Modality::Video));
        assert!(!spec.worker_expandable(Modality::Audio));
        assert_eq!(
            spec.video_frame_sampling(),
            FrameSampling::UpTo { max_frames: 32 }
        );
        assert_eq!(spec.default_video_sample_fps(), None);
        assert_eq!(spec.processor_kwargs(&metadata).unwrap(), json!({}));

        let image = spec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(image.encoder_input, FieldLayout::Batched);
        assert_eq!(
            image.model_specific,
            HashMap::from([("pixel_position_ids".to_string(), FieldLayout::Batched)])
        );
        let video = spec.encoder_field_layouts_for(Modality::Video);
        assert_eq!(video.encoder_input, FieldLayout::flat("video_frame_counts"));
        assert_eq!(
            video.model_specific,
            HashMap::from([
                (
                    "pixel_position_ids_videos".to_string(),
                    FieldLayout::flat("video_frame_counts")
                ),
                ("video_frame_counts".to_string(), FieldLayout::Batched),
            ])
        );
        assert_eq!(
            spec.encoder_input_key_for(Modality::Video).as_deref(),
            Some("pixel_values_videos")
        );
        assert_eq!(spec.encoder_input_key_for(Modality::Image), None);
        assert_eq!(
            spec.keep_on_cpu_keys_for(Modality::Video),
            vec!["video_frame_counts"]
        );
        assert!(spec.keep_on_cpu_keys_for(Modality::Image).is_empty());
        let legacy = spec.field_layouts();
        assert_eq!(legacy["pixel_values"], FieldLayout::Batched);
        assert_eq!(
            legacy["pixel_values_videos"],
            FieldLayout::flat("video_frame_counts")
        );
    }
}
