use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    media::FrameSampling,
    registry::{
        MediaItemInfo, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{
        EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId, VideoSamplingInfo,
    },
    vision::PreProcessorConfig,
};

pub(super) struct Qwen3VLVisionSpec;

impl Qwen3VLVisionSpec {
    fn image_pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }

    fn video_pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["video_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "video_token_id".to_string(),
            })
    }

    fn vision_start_token_id(metadata: &ModelMetadata) -> Option<TokenId> {
        metadata
            .config_u32(&["vision_start_token_id"])
            .map(|v| v as TokenId)
    }

    fn vision_end_token_id(metadata: &ModelMetadata) -> Option<TokenId> {
        metadata
            .config_u32(&["vision_end_token_id"])
            .map(|v| v as TokenId)
    }

    fn token_for_id(
        metadata: &ModelMetadata,
        token_id: TokenId,
        field: &str,
    ) -> RegistryResult<String> {
        metadata
            .tokenizer
            .id_to_token(token_id as u32)
            .ok_or_else(|| ModelRegistryError::TokenNotFound {
                token: format!("{field}:{token_id}"),
            })
    }

    /// `t` of media item `item`'s `video_grid_thw` row; `None` without a grid
    /// of one row per item.
    fn video_grid_t(preprocessed: &PreprocessedEncoderInputs, item: usize) -> Option<usize> {
        match preprocessed.model_specific.get("video_grid_thw") {
            Some(ModelSpecificValue::IntTensor { data, shape })
                if shape.len() == 2 && shape[1] == 3 && data.len() == shape[0] * 3 =>
            {
                usize::try_from(*data.get(item * 3)?).ok()
            }
            _ => None,
        }
    }

    /// Frames per temporal group: the preprocessor config's value, else the
    /// model config's, else the family's 2.
    fn temporal_patch_size(
        metadata: &ModelMetadata,
        preprocessor_config: Option<&PreProcessorConfig>,
    ) -> usize {
        preprocessor_config
            .and_then(|config| config.temporal_patch_size)
            .or_else(|| {
                metadata
                    .config_u32(&["vision_config", "temporal_patch_size"])
                    .map(|value| value as usize)
            })
            .unwrap_or(2)
            .max(1)
    }

    /// The rate the frames are taken to be sampled at when the decoder
    /// reported no timing: the preprocessor config's `fps`, else the family's 2.
    fn sample_fps(preprocessor_config: Option<&PreProcessorConfig>) -> f64 {
        preprocessor_config
            .and_then(|config| config.get_extra::<f32>("fps"))
            .filter(|fps| fps.is_finite() && *fps > 0.0)
            .map_or(2.0, f64::from)
    }

    /// The seconds stamped on each of `grid_t` temporal groups, as the family's
    /// reference processor computes them: every sampled frame's time is its
    /// source index over the source rate, a group is stamped with the mean of
    /// its first and last frame's time, and a trailing partial group repeats
    /// its last frame. Without the decoder's sampling the frames are taken as
    /// evenly spaced at `sample_fps`.
    fn group_timestamps(
        sampling: Option<&VideoSamplingInfo>,
        grid_t: usize,
        temporal_patch_size: usize,
        sample_fps: f64,
    ) -> Vec<f64> {
        let sampling = sampling.filter(|sampling| {
            sampling.source_fps.is_finite()
                && sampling.source_fps > 0.0
                && !sampling.frame_indices.is_empty()
        });
        match sampling {
            Some(sampling) => {
                let indices = &sampling.frame_indices;
                let last = indices[indices.len() - 1];
                let seconds = |index: usize| index as f64 / sampling.source_fps;
                (0..grid_t)
                    .map(|group| {
                        let first = group * temporal_patch_size;
                        let end = first + temporal_patch_size - 1;
                        let first = indices.get(first).copied().unwrap_or(last);
                        let end = indices.get(end).copied().unwrap_or(last);
                        (seconds(first) + seconds(end)) / 2.0
                    })
                    .collect()
            }
            None => (0..grid_t)
                .map(|group| {
                    (group as f64 * temporal_patch_size as f64
                        + (temporal_patch_size as f64 - 1.0) / 2.0)
                        / sample_fps
                })
                .collect(),
        }
    }

    fn encode_plain_text(metadata: &ModelMetadata, text: &str) -> Vec<TokenId> {
        metadata
            .tokenizer
            .encode_text(text)
            .map(|ids| ids.into_iter().map(|id| id as TokenId).collect())
            .unwrap_or_default()
    }

    /// The per-frame video placeholder body of the Qwen3-VL family: for every
    /// temporal group its `<seconds>` stamp, then `<|vision_start|>`, the
    /// group's pad tokens and `<|vision_end|>`, which is how the family's
    /// reference processor expands the bare `<|video_pad|>`; the chat
    /// template's outer `<|vision_start|>`/`<|vision_end|>` pair stays around
    /// it. Each stamp is encoded on its own, as the reference does, with one
    /// decimal place. Returns `None` when the layout cannot apply (no markers
    /// or ragged token counts), leaving the caller to fall back to a flat pad
    /// block.
    fn per_frame_video_tokens(
        metadata: &ModelMetadata,
        pad_token_id: TokenId,
        num_tokens: usize,
        timestamps: &[f64],
    ) -> Option<Vec<TokenId>> {
        let grid_t = timestamps.len();
        if grid_t == 0 || num_tokens == 0 || !num_tokens.is_multiple_of(grid_t) {
            return None;
        }
        let vision_start = Self::vision_start_token_id(metadata)?;
        let vision_end = Self::vision_end_token_id(metadata)?;
        let tokens_per_grid = num_tokens / grid_t;
        let mut tokens = Vec::with_capacity(num_tokens + grid_t * 10);
        for &seconds in timestamps {
            tokens.extend(Self::encode_plain_text(
                metadata,
                &format!("<{seconds:.1} seconds>"),
            ));
            tokens.push(vision_start);
            tokens.extend(std::iter::repeat_n(pad_token_id, tokens_per_grid));
            tokens.push(vision_end);
        }
        Some(tokens)
    }

    /// One replacement per video, stamped from each clip's sampling when the
    /// decoder reported it (`sampling[item]`), else at the assumed rate.
    fn video_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        sampling: &[Option<&VideoSamplingInfo>],
        preprocessor_config: Option<&PreProcessorConfig>,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::video_pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token_for(metadata, Modality::Video)?;
        let temporal_patch_size = Self::temporal_patch_size(metadata, preprocessor_config);
        let sample_fps = Self::sample_fps(preprocessor_config);
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .enumerate()
            .map(|(item, &num_tokens)| {
                // Every model routed to this spec (base VL and the 3.5/3.6
                // family) takes the per-frame layout; a flat pad block only
                // when the grid is unknown.
                let tokens = Self::video_grid_t(preprocessed, item)
                    .and_then(|grid_t| {
                        let timestamps = Self::group_timestamps(
                            sampling.get(item).copied().flatten(),
                            grid_t,
                            temporal_patch_size,
                            sample_fps,
                        );
                        Self::per_frame_video_tokens(
                            metadata,
                            pad_token_id,
                            num_tokens,
                            &timestamps,
                        )
                    })
                    .unwrap_or_else(|| vec![pad_token_id; num_tokens]);
                PromptReplacement::sequence(Modality::Video, &placeholder_token, tokens)
            })
            .collect())
    }
}

impl ModelProcessorSpec for Qwen3VLVisionSpec {
    fn name(&self) -> &'static str {
        "qwen3_vl"
    }

    /// `<|image_pad|>` / `<|video_pad|>` are vLLM's single-token targets.
    fn worker_expandable(&self, modality: Modality) -> bool {
        matches!(modality, Modality::Image | Modality::Video)
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        let model_type = metadata.config_model_type();
        let is_qwen3_vl = id.contains("qwen3") && id.contains("vl")
            || model_type.is_some_and(|mt| mt == "qwen3_vl");
        let is_qwen3_5 = id.contains("qwen3.5")
            || id.contains("qwen3.6")
            || model_type.is_some_and(|mt| mt == "qwen3_5" || mt == "qwen3_5_moe");
        let is_qwen4_exp = model_type.is_some_and(|mt| mt == "qwen4_exp");
        is_qwen3_vl || is_qwen3_5 || is_qwen4_exp
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        Self::token_for_id(
            metadata,
            Self::image_pad_token_id(metadata)?,
            "image_token_id",
        )
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_pad_token_id(metadata)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            Modality::Video => Self::token_for_id(
                metadata,
                Self::video_pad_token_id(metadata)?,
                "video_token_id",
            ),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image => Self::image_pad_token_id(metadata),
            Modality::Video => Self::video_pad_token_id(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        let mut limits = HashMap::from([(Modality::Image, 10)]);
        if metadata.config_u32(&["video_token_id"]).is_some() {
            limits.insert(Modality::Video, 1);
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
        let pad_token_id = Self::image_pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        // The chat template already wraps each image with <|vision_start|> ... <|vision_end|>,
        // so we only expand the single <|image_pad|> placeholder to N pad tokens.
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                let tokens = vec![pad_token_id; num_tokens];
                PromptReplacement::sequence(Modality::Image, &placeholder_token, tokens)
            })
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
            Modality::Video => self.video_replacements(metadata, preprocessed, &[], None),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    /// Video stamps come from each clip's sampled frames and source rate, as
    /// the family's reference processor computes them from the frame indices
    /// and the rate it is given.
    fn prompt_replacements_with_media(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
        media: &[MediaItemInfo],
        preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Video => {
                let sampling: Vec<Option<&VideoSamplingInfo>> = media
                    .iter()
                    .map(|item| item.video_sampling.as_ref())
                    .collect();
                self.video_replacements(
                    metadata,
                    preprocessed,
                    &sampling,
                    Some(preprocessor_config),
                )
            }
            _ => self.prompt_replacements_for(metadata, preprocessed, modality),
        }
    }

    /// The family's reference video processor picks `trunc(duration x 2)`
    /// frames, at least 4 and at most 768, at the nearest frames of an even
    /// spread; the prompt stamps carry those frames' times.
    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::EvenRounded
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // encoder_input is patchified: [total_patches, patch_features].
        // patches_per_image tells how many patches belong to each image.
        // image_grid_thw is [num_images, 3].
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_video".to_string(), FieldLayout::Batched),
            ("video_second_per_grid".to_string(), FieldLayout::Batched),
        ])
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        // One layout per modality. A video batch carries `patches_per_video`
        // and `video_grid_thw` and no image tensor at all, so slicing its
        // patches by `patches_per_image` (the legacy map's one answer) fails
        // at assembly for every video; the image batch is the mirror case.
        match modality {
            Modality::Video => EncoderFieldLayouts::new(
                FieldLayout::flat("patches_per_video"),
                HashMap::from([
                    ("video_grid_thw".to_string(), FieldLayout::Batched),
                    ("patches_per_video".to_string(), FieldLayout::Batched),
                    ("video_second_per_grid".to_string(), FieldLayout::Batched),
                ]),
            ),
            _ => EncoderFieldLayouts::new(
                FieldLayout::flat("patches_per_image"),
                HashMap::from([
                    ("image_grid_thw".to_string(), FieldLayout::Batched),
                    ("patches_per_image".to_string(), FieldLayout::Batched),
                ]),
            ),
        }
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        encoder_inputs::ModelSpecificValue,
        media::FrameSampling,
        registry::{test_helpers::*, MediaItemInfo, ModelMetadata, ModelRegistry},
        types::{FieldLayout, ImageSize, Modality, TokenId, VideoSamplingInfo},
        vision::PreProcessorConfig,
    };

    const VIDEO_PAD: TokenId = 151656;
    const VISION_START: TokenId = 151652;
    const VISION_END: TokenId = 151653;
    /// Where the byte-encoding test tokenizer puts plain text.
    const TEXT_BASE: u32 = 1000;

    fn stamp(text: &str) -> Vec<TokenId> {
        text.bytes()
            .map(|byte| (TEXT_BASE + u32::from(byte)) as TokenId)
            .collect()
    }

    /// One temporal group as the reference lays it out: the stamp, the
    /// opener, the pads, the closer.
    fn group(seconds: &str, pads: usize) -> Vec<TokenId> {
        let mut tokens = stamp(&format!("<{seconds} seconds>"));
        tokens.push(VISION_START);
        tokens.extend(std::iter::repeat_n(VIDEO_PAD, pads));
        tokens.push(VISION_END);
        tokens
    }

    #[test]
    fn qwen3_vl_pad_only_replacement() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999), ("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653,
            "vision_config": {"patch_size": 16}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen3-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3 spec");
        assert_eq!(spec.name(), "qwen3_vl");
        // 448/16 = 28 grid, merge_size=2 => (28*28)/4 = 196 tokens
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[196]),
            )
            .unwrap();
        // Only pad tokens — vision_start/vision_end are already in the chat template
        assert_eq!(replacements[0].tokens.len(), 196);
        assert_eq!(replacements[0].tokens[0], 151655); // pad (image_token_id)
        assert_eq!(*replacements[0].tokens.last().unwrap(), 151655); // pad
    }

    #[test]
    fn qwen3_vl_video_pad_replacement() {
        let tokenizer = TestTokenizer::new(&[("<|video_pad|>", 151656)]);
        let config = json!({
            "model_type": "qwen3_5",
            "image_token_id": 151655,
            "video_token_id": 151656,
        });
        let metadata = ModelMetadata {
            model_id: "Qwen3.5-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3.5 spec");
        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[128]),
                Modality::Video,
            )
            .unwrap();

        assert_eq!(replacements[0].modality, Modality::Video);
        assert_eq!(replacements[0].tokens.len(), 128);
        assert_eq!(replacements[0].tokens[0], 151656);
    }

    fn video_tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            ("<|video_pad|>", VIDEO_PAD as u32),
            ("<|vision_start|>", VISION_START as u32),
            ("<|vision_end|>", VISION_END as u32),
        ])
        .with_byte_encoder(TEXT_BASE)
    }

    fn video_config(model_type: &str) -> serde_json::Value {
        json!({
            "model_type": model_type,
            "image_token_id": 151655,
            "video_token_id": VIDEO_PAD,
            "vision_start_token_id": VISION_START,
            "vision_end_token_id": VISION_END,
            "vision_config": {"patch_size": 16, "temporal_patch_size": 2},
        })
    }

    /// Two temporal groups of 80 pads laid out as the reference processor
    /// expands the bare `<|video_pad|>`: every group stamped and wrapped in its
    /// own markers, the template's outer pair left in place. Without the
    /// decoder's timing the stamps assume the 2 fps sampling rate.
    fn assert_two_groups_at_the_assumed_rate(model_type: &str, model_id: &str) {
        let tokenizer = video_tokenizer();
        let config = video_config(model_type);
        let metadata = ModelMetadata {
            model_id,
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3 spec");
        assert_eq!(spec.name(), "qwen3_vl");
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(320, 256)], &[160])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![2, 16, 20], 1, 3),
            );
        let replacements = spec
            .prompt_replacements_for(&metadata, &preprocessed, Modality::Video)
            .unwrap();

        let mut expected = group("0.2", 80);
        expected.extend(group("1.2", 80));
        assert_eq!(replacements[0].tokens, expected);
        assert_eq!(replacements[0].structural_prefix, 0);
    }

    #[test]
    fn qwen3_5_video_replacement_splits_temporal_grid() {
        assert_two_groups_at_the_assumed_rate("qwen3_5", "Qwen/Qwen3.5-4B");
    }

    #[test]
    fn qwen3_vl_video_splits_temporal_grid() {
        // Base Qwen3-VL (not the 3.5/3.6 family) takes the same per-frame
        // layout: the engine's position pass expects one marker pair per
        // temporal group.
        assert_two_groups_at_the_assumed_rate("qwen3_vl", "Qwen/Qwen3-VL-8B-Instruct");
    }

    #[test]
    fn qwen3_vl_video_stamps_follow_the_sampled_frames() {
        // The decoder's sampling: a 1 s clip at 24 fps sampled at the four
        // frames the reference picks. The reference stamps each temporal pair
        // with the mean of its two frames' times: (0 + 8) / 2 / 24 = 0.17 and
        // (15 + 23) / 2 / 24 = 0.79 seconds.
        let tokenizer = video_tokenizer();
        let config = video_config("qwen3_vl");
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-VL-8B-Instruct",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3_vl spec");
        assert_eq!(spec.video_frame_sampling(), FrameSampling::EvenRounded);
        let preprocessor_config = PreProcessorConfig::from_json(
            r#"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,"fps":2}"#,
        )
        .expect("video preprocessor config");
        let media = [MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps: 24.0,
                frame_indices: vec![0, 8, 15, 23],
            }),
        }];
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(320, 256)], &[160])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![2, 16, 20], 1, 3),
            );
        let replacements = spec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                &media,
                &preprocessor_config,
            )
            .unwrap();
        let mut expected = group("0.2", 80);
        expected.extend(group("0.8", 80));
        assert_eq!(replacements[0].tokens, expected);

        // A trailing partial group repeats its last frame, as the reference
        // pads the indices: frames 0, 10, 20 at 10 fps give (0 + 10) / 2 / 10
        // and (20 + 20) / 2 / 10 seconds; a single group is laid out the same
        // way, not as a flat pad block.
        let media = [MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps: 10.0,
                frame_indices: vec![0, 10, 20],
            }),
        }];
        let replacements = spec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                &media,
                &preprocessor_config,
            )
            .unwrap();
        let mut expected = group("0.5", 80);
        expected.extend(group("2.0", 80));
        assert_eq!(replacements[0].tokens, expected);

        let one_group = test_preprocessed_with_tokens(&[ImageSize::new(320, 256)], &[80])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![1, 16, 20], 1, 3),
            );
        let replacements = spec
            .prompt_replacements_with_media(
                &metadata,
                &one_group,
                Modality::Video,
                &media,
                &preprocessor_config,
            )
            .unwrap();
        assert_eq!(replacements[0].tokens, group("0.5", 80));
    }

    #[test]
    fn qwen2_vl_does_not_match_qwen3() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653,
            "vision_config": {"patch_size": 16}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen3-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("should match qwen3");
        // Must match qwen3_vl spec, not qwen_vl
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen3_vl_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match qwen3 alias");
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen3_5_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen3_5_moe",
            "image_token_id": 151655,
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match qwen3.5 alias");
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen4_exp_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen4_exp",
            "image_token_id": 151655,
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match qwen4_exp alias");
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen3_vl_layouts_are_modality_specific() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "image_token_id": 151655,
            "video_token_id": 151656,
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3 spec");

        // A video batch carries patches_per_video / video_grid_thw and nothing
        // image-shaped: its patches are sliced by the sizes tensor it has.
        let video = spec.encoder_field_layouts_for(Modality::Video);
        assert_eq!(video.encoder_input, FieldLayout::flat("patches_per_video"));
        for key in [
            "video_grid_thw",
            "patches_per_video",
            "video_second_per_grid",
        ] {
            assert_eq!(
                video.model_specific.get(key),
                Some(&FieldLayout::Batched),
                "{key}"
            );
        }
        assert!(!video.model_specific.contains_key("patches_per_image"));
        assert!(!video.model_specific.contains_key("image_grid_thw"));

        let image = spec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(image.encoder_input, FieldLayout::flat("patches_per_image"));
        for key in ["image_grid_thw", "patches_per_image"] {
            assert_eq!(
                image.model_specific.get(key),
                Some(&FieldLayout::Batched),
                "{key}"
            );
        }
        assert!(!image.model_specific.contains_key("patches_per_video"));
    }
}
