//! Nemotron-H Omni prompt contract.
//!
//! Mirrors the serving engine's Nemotron-Omni processor. The chat template renders one
//! `<image>` per image, one `<so_embedding>` per audio and the literal text
//! `<video>` per video. An image's anchor becomes `<img>`, one `<image>` per
//! embedding of its patch grid after the 2x2 pixel shuffle, and `</img>`; an
//! audio's becomes `<so_start>`, one `<so_embedding>` per encoder position
//! over its 30-second clips, and `<so_end>`. A video becomes, per tubelet of
//! two frames, a separator naming the frames and their timestamps followed
//! by the tubelet's `<img>`, `<image>` run and `</img>`; the separator is
//! tokenized on its own, as the engine does. `<video>` is several tokens, so the
//! gateway rewrites it to a reserved, otherwise unused token as the anchor.
//! Only the `<image>` and `<so_embedding>` runs are feature positions.
//!
//! Frames are sampled as the engine's default loader samples them: every frame of
//! a clip with at most 32, else 32 spread evenly across the clip. A clip whose
//! decoder reported no frame rate or indices cannot be stamped and is refused.
//!
//! The processors hard-code this checkpoint family's geometry; a config that
//! differs is refused rather than served with the wrong grids.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    audio::{AudioPreProcessor, ParakeetAudioParams, ParakeetAudioProcessor, ParakeetConfigError},
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    media::FrameSampling,
    registry::{
        EncoderInputFormat, MediaItemInfo, ModelMetadata, ModelProcessorSpec, ModelRegistryError,
        RegistryResult,
    },
    types::{
        EncoderFieldLayouts, FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId,
        VideoSamplingInfo,
    },
    vision::{
        processors::nemotron_omni::{grid_tokens, VIDEO_TEMPORAL_PATCH_SIZE},
        NemotronOmniProcessor, PreProcessorConfig,
    },
};

const MODEL_TYPE: &str = "NemotronH_Nano_Omni_Reasoning_V3";
const IMAGE_PLACEHOLDER: &str = "<image>";
const AUDIO_PLACEHOLDER: &str = "<so_embedding>";
/// What the chat template writes for a video, and the reserved token the
/// gateway rewrites it to as the video's anchor.
const VIDEO_MARKER: &str = "<video>";
const VIDEO_ANCHOR: &str = "<SPECIAL_30>";
/// The engine's default video loader keeps at most this many frames of a clip.
const MAX_SAMPLED_FRAMES: usize = 32;
/// The engine leaves these counts unlimited; they are per-request defaults a
/// deployment changes with `SMG_IMAGE_MAX_COUNT` / `SMG_VIDEO_MAX_COUNT` /
/// `SMG_AUDIO_MAX_COUNT`.
const MAX_IMAGES_PER_REQUEST: usize = 16;
const MAX_AUDIOS_PER_REQUEST: usize = 4;
const MAX_VIDEOS_PER_REQUEST: usize = 2;

pub(super) struct NemotronOmniSpec;

impl NemotronOmniSpec {
    fn unsupported(modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: "nemotron_omni",
            modality,
        }
    }

    /// The anchor token, the feature token id and the wrapping begin/end ids.
    /// Video frames fill `<image>` positions.
    fn tokens(
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<(&'static str, TokenId, TokenId, TokenId)> {
        let (anchor, feature, begin, end) = match modality {
            Modality::Image => (IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER, "<img>", "</img>"),
            Modality::Video => (VIDEO_ANCHOR, IMAGE_PLACEHOLDER, "<img>", "</img>"),
            Modality::Audio => (
                AUDIO_PLACEHOLDER,
                AUDIO_PLACEHOLDER,
                "<so_start>",
                "<so_end>",
            ),
            Modality::ImageEmbeds => return Err(Self::unsupported(modality)),
        };
        Ok((
            anchor,
            metadata.token_id(feature)?,
            metadata.token_id(begin)?,
            metadata.token_id(end)?,
        ))
    }

    /// `(frames, rows, cols)` per video, as the paired processor emits them.
    fn video_grids(preprocessed: &PreprocessedEncoderInputs) -> RegistryResult<Vec<[usize; 3]>> {
        let invalid = || ModelRegistryError::InvalidPreprocessedField {
            field: "video_grid_thw".to_string(),
        };
        let Some(ModelSpecificValue::IntTensor { data, .. }) =
            preprocessed.model_specific.get("video_grid_thw")
        else {
            return Err(invalid());
        };
        if data.len() != 3 * preprocessed.feature_token_counts.len() {
            return Err(invalid());
        }
        data.chunks(3)
            .map(|grid| {
                let mut out = [0; 3];
                for (o, &v) in out.iter_mut().zip(grid) {
                    *o = usize::try_from(v)
                        .ok()
                        .filter(|v| *v > 0)
                        .ok_or_else(invalid)?;
                }
                Ok(out)
            })
            .collect()
    }

    /// The separator before each tubelet: its frames and their timestamps
    /// (the source index times `int(1000 / fps)` milliseconds).
    fn tubelet_separator(
        tubelet: usize,
        frames: usize,
        sampling: &VideoSamplingInfo,
    ) -> RegistryResult<String> {
        let first = tubelet * VIDEO_TEMPORAL_PATCH_SIZE;
        let frame_ms = (1000.0 / sampling.source_fps).trunc();
        let mut parts = Vec::with_capacity(VIDEO_TEMPORAL_PATCH_SIZE);
        for frame in first..(first + VIDEO_TEMPORAL_PATCH_SIZE).min(frames) {
            let index = sampling.frame_indices.get(frame).ok_or_else(|| {
                ModelRegistryError::InvalidPreprocessedField {
                    field: "video frame_indices".to_string(),
                }
            })?;
            let seconds = *index as f64 * frame_ms / 1000.0;
            let word = if frame == first { "Frame" } else { "frame" };
            parts.push(format!(
                "{word} {} sampled at {seconds:.2} seconds",
                frame + 1
            ));
        }
        let separator = format!("{}: ", parts.join(" and "));
        Ok(if tubelet > 0 {
            format!("\n{separator}")
        } else {
            separator
        })
    }

    fn video_replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        media: &[MediaItemInfo],
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let (anchor, token, begin, end) = Self::tokens(metadata, Modality::Video)?;
        Self::video_grids(preprocessed)?
            .into_iter()
            .enumerate()
            .map(|(item, [frames, rows, cols])| {
                let sampling = media
                    .get(item)
                    .and_then(|m| m.video_sampling.as_ref())
                    .filter(|s| s.source_fps.is_finite() && s.source_fps > 0.0)
                    .ok_or_else(|| ModelRegistryError::InvalidPreprocessedField {
                        field: "video frame times (the decoder reported no frame rate)".to_string(),
                    })?;
                let per_tubelet = grid_tokens(rows, cols);
                let mut tokens = Vec::new();
                let mut ranges = Vec::new();
                for tubelet in 0..frames.div_ceil(VIDEO_TEMPORAL_PATCH_SIZE) {
                    let separator = Self::tubelet_separator(tubelet, frames, sampling)?;
                    tokens.extend(metadata.encode_plain("nemotron_omni", &separator)?);
                    tokens.push(begin);
                    ranges.push(PlaceholderRange {
                        offset: tokens.len(),
                        length: per_tubelet,
                    });
                    tokens.extend(std::iter::repeat_n(token, per_tubelet));
                    tokens.push(end);
                }
                Ok(PromptReplacement::sequence(Modality::Video, anchor, tokens)
                    .with_feature_ranges(ranges))
            })
            .collect()
    }

    fn has_audio(config: &Value) -> bool {
        config.get("sound_config").is_some_and(|c| !c.is_null())
    }

    /// Refuses a checkpoint whose geometry differs from what the processors implement.
    fn check_config(config: &Value) -> RegistryResult<()> {
        if let Some(field) = NemotronOmniProcessor::mismatched_config_field(config) {
            return Err(ModelRegistryError::UnsupportedModel(format!(
                "Nemotron-H Omni config {field} differs from the gateway's preprocessing"
            )));
        }
        if !Self::has_audio(config) {
            return Ok(());
        }
        match ParakeetAudioParams::from_model_config(config) {
            Ok(_) => Ok(()),
            Err(ParakeetConfigError::Missing(field)) => {
                Err(ModelRegistryError::MissingConfigField {
                    field: format!("sound_config.{field}"),
                })
            }
            Err(ParakeetConfigError::Unsupported(field)) => {
                Err(ModelRegistryError::UnsupportedModel(format!(
                    "Nemotron-H Omni config sound_config.{field} differs from the gateway's \
                     preprocessing"
                )))
            }
        }
    }
}

impl ModelProcessorSpec for NemotronOmniSpec {
    fn name(&self) -> &'static str {
        "nemotron_omni"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.config_model_type() == Some(MODEL_TYPE)
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_PLACEHOLDER.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(IMAGE_PLACEHOLDER)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        Ok(Self::tokens(metadata, modality)?.0.to_string())
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        Ok(Self::tokens(metadata, modality)?.1)
    }

    /// The processors emit packed patch rows and mel frames, not the engine's
    /// `pixel_values_flat` / `pixel_values_flat_video` inputs.
    fn encoder_input_format(&self) -> EncoderInputFormat {
        EncoderInputFormat::Packed
    }

    fn rendered_marker_for(&self, modality: Modality) -> Option<&'static str> {
        (modality == Modality::Video).then_some(VIDEO_MARKER)
    }

    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::UpTo {
            max_frames: MAX_SAMPLED_FRAMES,
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Self::check_config(metadata.config)?;
        let mut limits = HashMap::from([
            (Modality::Image, MAX_IMAGES_PER_REQUEST),
            (Modality::Video, MAX_VIDEOS_PER_REQUEST),
        ]);
        if Self::has_audio(metadata.config) {
            limits.insert(Modality::Audio, MAX_AUDIOS_PER_REQUEST);
        }
        Ok(limits)
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn audio_processor(
        &self,
        model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
    ) -> Option<Box<dyn AudioPreProcessor>> {
        let params = ParakeetAudioParams::from_model_config(model_config).ok()?;
        Some(Box::new(ParakeetAudioProcessor::new(params)))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.prompt_replacements_for(metadata, preprocessed, Modality::Image)
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
            Modality::Video => Self::video_replacements(metadata, preprocessed, media),
            _ => self.prompt_replacements_for(metadata, preprocessed, modality),
        }
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        if modality == Modality::Video {
            return Self::video_replacements(metadata, preprocessed, &[]);
        }
        let (placeholder, token, begin, end) = Self::tokens(metadata, modality)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                let mut tokens = Vec::with_capacity(count + 2);
                tokens.push(begin);
                tokens.extend(std::iter::repeat_n(token, count));
                tokens.push(end);
                PromptReplacement::sequence(modality, placeholder, tokens)
                    .with_feature_span(1, count)
            })
            .collect())
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Image => EncoderFieldLayouts::new(
                FieldLayout::flat("patches_per_image"),
                HashMap::from([
                    ("patches_per_image".to_string(), FieldLayout::Batched),
                    ("image_grid_hw".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::Audio => EncoderFieldLayouts::new(
                FieldLayout::flat("frames_per_item"),
                HashMap::from([
                    ("frames_per_item".to_string(), FieldLayout::Batched),
                    ("clips_per_item".to_string(), FieldLayout::Batched),
                    (
                        "audio_clip_frames".to_string(),
                        FieldLayout::flat("clips_per_item"),
                    ),
                ]),
            ),
            Modality::Video => EncoderFieldLayouts::new(
                FieldLayout::flat("patches_per_video"),
                HashMap::from([
                    ("patches_per_video".to_string(), FieldLayout::Batched),
                    ("video_grid_thw".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::ImageEmbeds => EncoderFieldLayouts::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::test_helpers::{test_preprocessed_with_tokens, TestTokenizer},
        types::ImageSize,
    };

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            ("<image>", 18),
            ("<img>", 19),
            ("</img>", 20),
            ("<so_embedding>", 27),
            ("<so_start>", 28),
            ("<so_end>", 29),
        ])
    }

    /// The checkpoint config fields the spec reads.
    fn checkpoint_config() -> Value {
        json!({
            "model_type": MODEL_TYPE,
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "norm_mean": [0.48145466, 0.4578275, 0.40821073],
            "norm_std": [0.26862954, 0.26130258, 0.27577711],
            "vision_config": {
                "args": {"min_num_patches": 1024, "max_num_patches": 13312},
                "video_target_num_patches": 1024,
                "video_temporal_patch_size": 2,
                "video_maintain_aspect_ratio": true
            },
            "sound_config": {"sampling_rate": 16000, "num_mel_bins": 128, "subsampling_factor": 8}
        })
    }

    #[test]
    fn refuses_a_config_off_the_implemented_geometry() {
        let tokenizer = tokenizer();
        let spec = NemotronOmniSpec;
        let limits = |config: &Value| {
            spec.modality_limits(&ModelMetadata {
                model_id: "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
                tokenizer: &tokenizer,
                config,
            })
        };
        assert!(limits(&checkpoint_config()).is_ok());
        // Numbers are compared by value, whatever their JSON type.
        let mut retyped = checkpoint_config();
        retyped["patch_size"] = json!(16.0);
        retyped["sound_config"]["padding_value"] = json!(0);
        retyped["sound_config"]["subsampling_conv_stride"] = json!(2.0);
        assert!(limits(&retyped).is_ok());
        // A checkpoint without the sound encoder serves images and video only.
        let mut vision_only = checkpoint_config();
        vision_only["sound_config"] = Value::Null;
        let vision_limits = limits(&vision_only).expect("no sound_config to check");
        assert!(!vision_limits.contains_key(&Modality::Audio));
        for (pointer, value) in [
            ("/patch_size", json!(14)),
            ("/downsample_ratio", json!(0.25)),
            ("/norm_mean", json!([0.5, 0.5, 0.5])),
            ("/norm_std", json!([0.5, 0.5, 0.5])),
            ("/vision_config/args/min_num_patches", json!(256)),
            ("/vision_config/args/max_num_patches", json!(4096)),
            ("/vision_config/video_target_num_patches", json!(256)),
            ("/vision_config/video_temporal_patch_size", json!(1)),
            ("/vision_config/video_maintain_aspect_ratio", json!(false)),
            ("/sound_config/subsampling_factor", json!(4)),
            ("/sound_config/hop_length", json!(320)),
            ("/sound_config/subsampling_conv_stride", json!(4)),
        ] {
            let mut config = checkpoint_config();
            let (parent, field) = pointer.rsplit_once('/').unwrap();
            config.pointer_mut(parent).unwrap()[field] = value;
            let error = limits(&config).unwrap_err().to_string();
            assert!(error.contains(field), "{pointer}: {error}");
        }
        let mut sized = checkpoint_config();
        sized["vision_config"]["video_target_img_size"] = json!(512);
        assert!(limits(&sized).is_err());
        let mut no_mels = checkpoint_config();
        no_mels["sound_config"]
            .as_object_mut()
            .unwrap()
            .remove("num_mel_bins");
        assert_eq!(
            limits(&no_mels).unwrap_err(),
            ModelRegistryError::MissingConfigField {
                field: "sound_config.num_mel_bins".to_string()
            }
        );
    }

    #[test]
    fn preprocesses_packed_inputs_with_the_reference_loader_sampling() {
        let spec = NemotronOmniSpec;
        assert_eq!(spec.encoder_input_format(), EncoderInputFormat::Packed);
        assert_eq!(
            spec.video_frame_sampling(),
            FrameSampling::UpTo { max_frames: 32 }
        );
        assert_eq!(spec.rendered_marker_for(Modality::Video), Some("<video>"));
    }

    #[test]
    fn wraps_images_and_audio_around_their_feature_runs() {
        let tokenizer = tokenizer();
        let config = checkpoint_config();
        let metadata = ModelMetadata {
            model_id: "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = NemotronOmniSpec;
        assert!(spec.matches(&metadata));
        let limits = spec.modality_limits(&metadata).unwrap();
        assert!(limits.contains_key(&Modality::Image) && limits.contains_key(&Modality::Audio));

        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(64, 32)], &[3]);
        for (modality, tokens) in [
            (Modality::Image, vec![19, 18, 18, 18, 20]),
            (Modality::Audio, vec![28, 27, 27, 27, 29]),
        ] {
            let replacements = spec
                .prompt_replacements_for(&metadata, &preprocessed, modality)
                .unwrap();
            assert_eq!(replacements.len(), 1);
            assert_eq!(replacements[0].tokens, tokens);
        }
    }

    #[test]
    fn stamps_each_video_tubelet_with_its_frames() {
        const TEXT_BASE: TokenId = 1000;
        let tokenizer = tokenizer().with_byte_encoder(TEXT_BASE.unsigned_abs());
        let config = checkpoint_config();
        let metadata = ModelMetadata {
            model_id: "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = NemotronOmniSpec;
        assert_eq!(spec.rendered_marker_for(Modality::Video), Some("<video>"));
        assert_eq!(spec.rendered_marker_for(Modality::Image), None);

        // Three frames of a 4x4 patch grid: a full tubelet, then a lone last frame.
        let mut preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(64, 64)], &[8]);
        preprocessed.model_specific.insert(
            "video_grid_thw".to_string(),
            ModelSpecificValue::IntTensor {
                data: vec![3, 4, 4],
                shape: vec![1, 3],
            },
        );
        let text = |s: &str| {
            s.bytes()
                .map(|b| TEXT_BASE + TokenId::from(b))
                .collect::<Vec<_>>()
        };
        let tubelet = [vec![19], vec![18; 4], vec![20]].concat();
        let sampled = |source_fps: f64, frame_indices: Vec<usize>| {
            vec![MediaItemInfo {
                video_sampling: Some(VideoSamplingInfo {
                    source_fps,
                    frame_indices,
                }),
            }]
        };
        let replace = |media: &[MediaItemInfo]| {
            spec.prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                media,
                &PreProcessorConfig::default(),
            )
        };
        // A clip the decoder could not report frame times for cannot be stamped.
        assert!(replace(&[]).is_err());
        assert!(replace(&sampled(0.0, vec![0, 1, 2])).is_err());
        for (media, first, second) in [
            // 33 ms per source frame: 15 * 33 ms = 0.495 s prints as 0.49, as Python's :.2f does.
            (
                sampled(30.0, vec![0, 15, 29]),
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.49 seconds: ",
                "\nFrame 3 sampled at 0.96 seconds: ",
            ),
            // int(1000 / 24) truncates to 41 ms.
            (
                sampled(24.0, vec![0, 720, 1440]),
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 29.52 seconds: ",
                "\nFrame 3 sampled at 59.04 seconds: ",
            ),
        ] {
            let replacements = replace(&media).unwrap();
            assert_eq!(replacements.len(), 1);
            let expected = [text(first), tubelet.clone(), text(second), tubelet.clone()].concat();
            assert_eq!(replacements[0].tokens, expected);
            assert_eq!(replacements[0].placeholder_token, "<SPECIAL_30>");
            let offsets: Vec<(usize, usize)> = replacements[0]
                .feature_ranges
                .iter()
                .flatten()
                .map(|range| (range.offset, range.length))
                .collect();
            let second_offset = first.len() + tubelet.len() + second.len() + 1;
            assert_eq!(offsets, [(first.len() + 1, 4), (second_offset, 4)]);
        }
    }
}
