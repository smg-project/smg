//! Nemotron VL / Omni prompt contract.
//!
//! Mirrors the serving engine's `NanoNemotronVLMultiModalProcessor`, which
//! serves the family's architectures (`ARCHITECTURES`) from their config. The
//! chat template renders one `<image>` per image and the literal text `<video>` per video:
//!
//! - an image's anchor becomes `<img>`, one `<image>` per embedding of its
//!   tiles or patch grid, and `</img>`;
//! - a video becomes, per tubelet of `video_temporal_patch_size` frames, a
//!   separator naming the frames and their timestamps, tokenized on its own as
//!   the engine does, followed by the tubelet's `<img>`, `<image>` run and
//!   `</img>`. `<video>` is several tokens, so the caller rewrites it to a
//!   reserved, otherwise unused token as the anchor. The engine embeds the
//!   whole replacement (it embeds the separators itself), so the whole
//!   replacement is the feature range.
//!
//! The tensors are the engine's reference names: `pixel_values_flat` (tiles
//! with `image_num_patches`, or one image per item with `imgs_sizes` and
//! `num_tokens_per_image`), and `pixel_values_flat_video` with
//! `video_num_patches`, `frames_indices` and `frame_duration_ms`. Tiled and
//! dynamic-resolution images lay their pixels out differently, so one spec
//! instance serves each. Audio is not served.
//!
//! Frames are sampled as the engine's default loader samples them: every
//! frame of a clip with at most 32, else 32 spread evenly across the clip.

use std::collections::HashMap;

use serde_json::Value;

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    media::FrameSampling,
    registry::{
        ModalityHooks, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{
        EncoderFieldLayouts, FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId,
    },
    vision::{
        processors::nemotron_vl::{
            NemotronVLConfigError, NemotronVLGeometry, DYNAMIC_IMAGE_SIDE_TENSORS,
        },
        NemotronVLProcessor, PreProcessorConfig, VisionPreProcessor,
    },
};

/// The engine's architectures for this processor.
const ARCHITECTURES: [&str; 4] = [
    "NemotronH_Nano_VL_V2",
    "NemotronH_Nano_Omni_Reasoning_V3",
    "NemotronH_Super_Omni_Reasoning_V3",
    "NemotronH_Omni_Reasoning_V3",
];
const IMAGE_PLACEHOLDER: &str = "<image>";
/// What the chat template writes for a video, and the reserved token the
/// caller rewrites it to as the video's anchor.
const VIDEO_MARKER: &str = "<video>";
const VIDEO_ANCHOR: &str = "<SPECIAL_30>";
/// The engine's default video loader keeps at most this many frames of a clip.
const MAX_SAMPLED_FRAMES: usize = 32;
/// A video's tensors beside its frames, one entry per video.
const VIDEO_SIDE_TENSORS: [&str; 3] = ["video_num_patches", "frames_indices", "frame_duration_ms"];

/// The Nemotron VL / Omni spec for checkpoints whose images are tiled
/// (`dynamic_images: false`) or sized dynamically (`true`).
pub(super) struct NemotronVLSpec {
    pub(super) dynamic_images: bool,
}

impl NemotronVLSpec {
    /// The feature token id and the wrapping begin/end ids. Video frames
    /// fill `<image>` positions too.
    fn tokens(metadata: &ModelMetadata) -> RegistryResult<(TokenId, TokenId, TokenId)> {
        Ok((
            metadata.token_id(IMAGE_PLACEHOLDER)?,
            metadata.token_id("<img>")?,
            metadata.token_id("</img>")?,
        ))
    }

    fn geometry(config: &Value) -> RegistryResult<NemotronVLGeometry> {
        let name = |ptr: &str| ptr.trim_start_matches('/').replace('/', ".");
        NemotronVLGeometry::from_model_config(config).map_err(|error| match error {
            NemotronVLConfigError::Missing(ptr) => {
                ModelRegistryError::MissingConfigField { field: name(ptr) }
            }
            NemotronVLConfigError::Invalid(ptr) => ModelRegistryError::InvalidConfigField {
                field: name(ptr),
                reason: "the engine's processor refuses this value".to_string(),
            },
        })
    }

    fn int_values<'a>(
        preprocessed: &'a PreprocessedEncoderInputs,
        key: &str,
    ) -> RegistryResult<&'a [i64]> {
        match preprocessed.model_specific.get(key) {
            Some(ModelSpecificValue::IntTensor { data, .. }) => Ok(data),
            _ => Err(ModelRegistryError::InvalidPreprocessedField {
                field: key.to_string(),
            }),
        }
    }

    /// The separator before tubelet `tubelet` of `frames` frames: its frames
    /// and their timestamps (the source index times `frame_ms` milliseconds,
    /// printed as Python's `:.2f` prints them).
    fn tubelet_separator(
        tubelet: usize,
        size: usize,
        frames: usize,
        indices: &[i64],
        frame_ms: i64,
    ) -> String {
        let first = tubelet * size;
        let parts: Vec<String> = (first..(first + size).min(frames))
            .map(|frame| {
                let seconds = indices[frame] as f64 * frame_ms as f64 / 1000.0;
                let word = if frame == first { "Frame" } else { "frame" };
                format!("{word} {} sampled at {seconds:.2} seconds", frame + 1)
            })
            .collect();
        let separator = format!("{}: ", parts.join(" and "));
        if tubelet > 0 {
            format!("\n{separator}")
        } else {
            separator
        }
    }

    fn image_replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let (token, begin, end) = Self::tokens(metadata)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                let mut tokens = Vec::with_capacity(count + 2);
                tokens.push(begin);
                tokens.extend(std::iter::repeat_n(token, count));
                tokens.push(end);
                PromptReplacement::sequence(Modality::Image, IMAGE_PLACEHOLDER, tokens)
                    .with_feature_span(1, count)
            })
            .collect())
    }

    fn video_replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let (token, begin, end) = Self::tokens(metadata)?;
        let geometry = Self::geometry(metadata.config)?;
        let size = geometry.video_temporal_patch_size.max(1);
        let frames_per_video = Self::int_values(preprocessed, "video_num_patches")?;
        let durations = Self::int_values(preprocessed, "frame_duration_ms")?;
        let mut indices = Self::int_values(preprocessed, "frames_indices")?;
        let invalid = |field: &str| ModelRegistryError::InvalidPreprocessedField {
            field: field.to_string(),
        };
        if frames_per_video.len() != preprocessed.feature_token_counts.len()
            || durations.len() != frames_per_video.len()
        {
            return Err(invalid("video_num_patches"));
        }
        let mut replacements = Vec::with_capacity(frames_per_video.len());
        for ((&frames, &frame_ms), &count) in frames_per_video
            .iter()
            .zip(durations)
            .zip(&preprocessed.feature_token_counts)
        {
            let frames = usize::try_from(frames)
                .ok()
                .filter(|&f| f > 0 && f <= indices.len())
                .ok_or_else(|| invalid("frames_indices"))?;
            let (own, rest) = indices.split_at(frames);
            indices = rest;
            let tubelets = geometry.tubelets(frames);
            let per_tubelet = count / tubelets;
            let mut tokens = Vec::new();
            for tubelet in 0..tubelets {
                let separator = Self::tubelet_separator(tubelet, size, frames, own, frame_ms);
                tokens.extend(metadata.encode_plain("nemotron_vl", &separator)?);
                tokens.push(begin);
                tokens.extend(std::iter::repeat_n(token, per_tubelet));
                tokens.push(end);
            }
            let length = tokens.len();
            replacements.push(
                PromptReplacement::sequence(Modality::Video, VIDEO_ANCHOR, tokens)
                    .with_feature_ranges(vec![PlaceholderRange { offset: 0, length }]),
            );
        }
        Ok(replacements)
    }
}

impl ModelProcessorSpec for NemotronVLSpec {
    fn name(&self) -> &'static str {
        if self.dynamic_images {
            "nemotron_vl"
        } else {
            "nemotron_vl_tiled"
        }
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let config = metadata.config;
        let named = |name: &str| ARCHITECTURES.contains(&name);
        let family = metadata.config_model_type().is_some_and(named)
            || config
                .get("architectures")
                .and_then(Value::as_array)
                .is_some_and(|names| names.iter().filter_map(Value::as_str).any(named));
        family
            && config
                .pointer("/vision_config/args/min_num_patches")
                .is_some_and(|v| !v.is_null())
                == self.dynamic_images
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_PLACEHOLDER.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(IMAGE_PLACEHOLDER)
    }

    fn rendered_marker_for(&self, modality: Modality) -> Option<&'static str> {
        (modality == Modality::Video).then_some(VIDEO_MARKER)
    }

    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::UpTo {
            max_frames: MAX_SAMPLED_FRAMES,
        }
    }

    /// The engine names no per-request count for the family, so neither does
    /// the spec.
    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Self::geometry(metadata.config)?;
        Ok(HashMap::from([
            (Modality::Image, usize::MAX),
            (Modality::Video, usize::MAX),
        ]))
    }

    fn vision_processor(
        &self,
        model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image | Modality::Video => Ok(Box::new(NemotronVLProcessor::new(
                Self::geometry(model_config)?,
            ))),
            _ => Err(ModelRegistryError::unsupported(self.name(), modality)),
        }
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        Self::image_replacements(metadata, preprocessed)
    }

    /// Images and video; the image layout is this instance's (tiled or
    /// dynamic).
    fn hooks(&self, modality: Modality) -> Option<ModalityHooks> {
        let image = |layouts, keep_on_cpu| ModalityHooks {
            placeholder: |_| Ok(IMAGE_PLACEHOLDER.to_string()),
            placeholder_id: |metadata| metadata.token_id(IMAGE_PLACEHOLDER),
            replacements: |metadata, preprocessed, _, _| {
                Self::image_replacements(metadata, preprocessed)
            },
            layouts,
            keep_on_cpu,
            encoder_input_key: Some("pixel_values_flat"),
            worker_expandable: false,
        };
        Some(match modality {
            Modality::Image if self.dynamic_images => image(
                || {
                    EncoderFieldLayouts::new(
                        FieldLayout::Batched,
                        HashMap::from(
                            DYNAMIC_IMAGE_SIDE_TENSORS
                                .map(|key| (key.to_string(), FieldLayout::Batched)),
                        ),
                    )
                },
                &DYNAMIC_IMAGE_SIDE_TENSORS,
            ),
            Modality::Image => image(
                || {
                    EncoderFieldLayouts::new(
                        FieldLayout::flat("image_num_patches"),
                        HashMap::from([("image_num_patches".to_string(), FieldLayout::Batched)]),
                    )
                },
                &["image_num_patches"],
            ),
            Modality::Video => ModalityHooks {
                placeholder: |_| Ok(VIDEO_ANCHOR.to_string()),
                // Video frames fill `<image>` positions too.
                placeholder_id: |metadata| metadata.token_id(IMAGE_PLACEHOLDER),
                replacements: |metadata, preprocessed, _, _| {
                    Self::video_replacements(metadata, preprocessed)
                },
                layouts: || {
                    EncoderFieldLayouts::new(
                        FieldLayout::flat("video_num_patches"),
                        HashMap::from(
                            VIDEO_SIDE_TENSORS.map(|key| (key.to_string(), FieldLayout::Batched)),
                        ),
                    )
                },
                keep_on_cpu: &VIDEO_SIDE_TENSORS,
                encoder_input_key: Some("pixel_values_flat_video"),
                worker_expandable: false,
            },
            Modality::Audio | Modality::ImageEmbeds => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::{
            test_helpers::{test_preprocessed_with_tokens, TestTokenizer},
            EncoderInputFormat, ModelRegistry,
        },
        types::ImageSize,
    };

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            ("<image>", 18),
            ("<img>", 19),
            ("</img>", 20),
            ("<SPECIAL_30>", 30),
        ])
    }

    /// The fields of the Nemotron 3 Nano Omni config the spec reads.
    fn omni_config() -> Value {
        json!({
            "architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
            "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "force_image_size": 512,
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

    /// The Nemotron Nano V2 VL config: tiles, frames one at a time.
    fn v2_config() -> Value {
        let mut config = omni_config();
        config["architectures"] = json!(["NemotronH_Nano_VL_V2"]);
        config["model_type"] = json!("NemotronH_Nano_VL_V2");
        config["use_thumbnail"] = json!(true);
        config["vision_config"] = json!({"args": {}});
        config.as_object_mut().unwrap().remove("sound_config");
        config
    }

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "nvidia/test",
            tokenizer,
            config,
        }
    }

    #[test]
    fn serves_each_architecture_by_its_image_layout() {
        let tokenizer = tokenizer();
        let tiled = NemotronVLSpec {
            dynamic_images: false,
        };
        let dynamic = NemotronVLSpec {
            dynamic_images: true,
        };
        let omni = omni_config();
        let v2 = v2_config();
        // NemotronH_Omni_Reasoning_V3: its own model_type, the architecture names it.
        let mut super_vl = omni_config();
        super_vl["model_type"] = json!("nemotron_h_omni");
        super_vl["architectures"] = json!(["NemotronH_Omni_Reasoning_V3"]);
        super_vl["sound_config"] = Value::Null;
        for (config, is_dynamic) in [(&omni, true), (&v2, false), (&super_vl, true)] {
            let metadata = metadata(&tokenizer, config);
            assert_eq!(dynamic.matches(&metadata), is_dynamic);
            assert_eq!(tiled.matches(&metadata), !is_dynamic);
            // Images and video; the Omni checkpoint's audio is not served.
            let limits = if is_dynamic { &dynamic } else { &tiled }
                .modality_limits(&metadata)
                .unwrap();
            assert_eq!(limits.len(), 2);
            assert!(!limits.contains_key(&Modality::Audio));
        }
        let other = json!({"model_type": "qwen3_vl", "vision_config": {"args": {}}});
        assert!(!tiled.matches(&metadata(&tokenizer, &other)));

        let mut broken = omni_config();
        broken.as_object_mut().unwrap().remove("patch_size");
        assert_eq!(
            dynamic.modality_limits(&metadata(&tokenizer, &broken)),
            Err(ModelRegistryError::MissingConfigField {
                field: "patch_size".to_string()
            })
        );
        broken["patch_size"] = json!(16);
        broken["downsample_ratio"] = json!(2);
        assert!(matches!(
            dynamic.modality_limits(&metadata(&tokenizer, &broken)),
            Err(ModelRegistryError::InvalidConfigField { field, .. }) if field == "downsample_ratio"
        ));
    }

    #[test]
    fn only_a_reference_registry_serves_the_reference_layout() {
        let tokenizer = tokenizer();
        let (omni, v2) = (omni_config(), v2_config());
        let name = |registry: &ModelRegistry, config: &Value| {
            registry
                .lookup(&metadata(&tokenizer, config))
                .map(|spec| spec.name())
        };
        let default = ModelRegistry::new();
        let reference = ModelRegistry::for_format(EncoderInputFormat::Reference);
        assert_eq!(name(&default, &omni), Some("nemotron_omni"));
        assert_eq!(name(&default, &v2), None);
        assert_eq!(name(&reference, &omni), Some("nemotron_vl"));
        assert_eq!(name(&reference, &v2), Some("nemotron_vl_tiled"));
        let packed = ModelRegistry::for_format(EncoderInputFormat::Packed);
        assert_eq!(name(&packed, &omni), Some("nemotron_omni"));
        assert_eq!(name(&packed, &v2), None);

        let serves = |registry: &ModelRegistry, spec| registry.spec_names().contains(&spec);
        assert!(!serves(&default, "nemotron_vl"));
        assert!(!serves(&reference, "nemotron_omni"));
        assert!(serves(&reference, "qwen3_vl") && !serves(&packed, "qwen3_vl"));
    }

    #[test]
    fn names_the_engines_tensors() {
        let tiled = NemotronVLSpec {
            dynamic_images: false,
        };
        let dynamic = NemotronVLSpec {
            dynamic_images: true,
        };
        assert_eq!(
            dynamic.encoder_input_format(),
            EncoderInputFormat::Reference
        );
        assert_eq!(
            dynamic.encoder_input_key_for(Modality::Image).as_deref(),
            Some("pixel_values_flat")
        );
        assert_eq!(
            dynamic.encoder_input_key_for(Modality::Video).as_deref(),
            Some("pixel_values_flat_video")
        );
        assert_eq!(
            tiled
                .encoder_field_layouts_for(Modality::Image)
                .encoder_input,
            FieldLayout::flat("image_num_patches")
        );
        assert_eq!(
            dynamic
                .encoder_field_layouts_for(Modality::Image)
                .encoder_input,
            FieldLayout::Batched
        );
        assert_eq!(
            dynamic.keep_on_cpu_keys_for(Modality::Video),
            ["video_num_patches", "frames_indices", "frame_duration_ms"]
        );
        assert_eq!(
            dynamic.rendered_marker_for(Modality::Video),
            Some("<video>")
        );
    }

    #[test]
    fn wraps_images_around_their_feature_runs() {
        let tokenizer = tokenizer();
        let config = omni_config();
        let metadata = metadata(&tokenizer, &config);
        let spec = NemotronVLSpec {
            dynamic_images: true,
        };
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(64, 32)], &[3]);
        let replacements = spec
            .prompt_replacements_for(&metadata, &preprocessed, Modality::Image)
            .unwrap();
        assert_eq!(replacements.len(), 1);
        assert_eq!(replacements[0].tokens, vec![19, 18, 18, 18, 20]);
        assert!(spec
            .prompt_replacements_for(&metadata, &preprocessed, Modality::Audio)
            .is_err());
    }

    fn video(frames: &[i64], frame_ms: i64, tokens: usize) -> PreprocessedEncoderInputs {
        let mut preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(64, 64)], &[tokens]);
        for (key, value) in [
            (
                "video_num_patches",
                ModelSpecificValue::int_1d(vec![frames.len() as i64]),
            ),
            (
                "frame_duration_ms",
                ModelSpecificValue::int_1d(vec![frame_ms]),
            ),
            (
                "frames_indices",
                ModelSpecificValue::int_2d(frames.to_vec(), 1, frames.len()),
            ),
        ] {
            preprocessed.model_specific.insert(key.to_string(), value);
        }
        preprocessed
    }

    #[test]
    fn stamps_each_video_tubelet_with_its_frames() {
        const TEXT_BASE: TokenId = 1000;
        let tokenizer = tokenizer().with_byte_encoder(TEXT_BASE.unsigned_abs());
        let text = |s: &str| {
            s.bytes()
                .map(|b| TEXT_BASE + TokenId::from(b))
                .collect::<Vec<_>>()
        };
        let spec = NemotronVLSpec {
            dynamic_images: true,
        };
        let tubelet = [vec![19], vec![18; 4], vec![20]].concat();
        let omni = omni_config();
        for (frames, frame_ms, first, second) in [
            // 33 ms per source frame: 15 * 33 ms = 0.495 s prints as 0.49, as Python's :.2f does.
            (
                [0, 15, 29],
                33,
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.49 seconds: ",
                "\nFrame 3 sampled at 0.96 seconds: ",
            ),
            // int(1000 / 24) truncates to 41 ms.
            (
                [0, 720, 1440],
                41,
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 29.52 seconds: ",
                "\nFrame 3 sampled at 59.04 seconds: ",
            ),
        ] {
            let replacements = spec
                .prompt_replacements_for(
                    &metadata(&tokenizer, &omni),
                    &video(&frames, frame_ms, 8),
                    Modality::Video,
                )
                .unwrap();
            assert_eq!(replacements.len(), 1);
            let expected = [text(first), tubelet.clone(), text(second), tubelet.clone()].concat();
            assert_eq!(replacements[0].tokens, expected);
            assert_eq!(replacements[0].placeholder_token, "<SPECIAL_30>");
            // The engine embeds the whole replacement.
            let ranges: Vec<_> = replacements[0].feature_ranges.iter().flatten().collect();
            assert_eq!(
                ranges,
                [&PlaceholderRange {
                    offset: 0,
                    length: expected.len()
                }]
            );
        }

        // V2 stamps every frame on its own.
        let v2 = v2_config();
        let replacements = spec
            .prompt_replacements_for(
                &metadata(&tokenizer, &v2),
                &video(&[0, 10], 100, 8),
                Modality::Video,
            )
            .unwrap();
        let expected = [
            text("Frame 1 sampled at 0.00 seconds: "),
            tubelet.clone(),
            text("\nFrame 2 sampled at 1.00 seconds: "),
            tubelet,
        ]
        .concat();
        assert_eq!(replacements[0].tokens, expected);
    }
}
