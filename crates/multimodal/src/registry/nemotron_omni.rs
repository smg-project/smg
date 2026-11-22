//! Nemotron VL / Omni prompt contract.
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
//! The processors hard-code this checkpoint's geometry; a config that
//! differs is refused rather than served with the wrong grids.
//!
//! That is the `Packed` instance (`nemotron_omni`): Nemotron 3 Nano Omni as
//! packed patch rows and mel frames. The `Reference` (`nemotron_vl`) and
//! `ReferenceTiled` (`nemotron_vl_tiled`) instances serve the family's
//! architectures (`ARCHITECTURES`) from their config in the engine's own
//! tensors, without audio: `pixel_values_flat` (dynamic-resolution
//! `(patches, 3 * p * p)` rows or tiled `(tiles, 3, S, S)` pixels with
//! `image_num_patches`, dynamic-resolution images additionally carrying
//! `imgs_sizes` and `num_tokens_per_image`; the two resize their images
//! differently, so each takes an instance) and `pixel_values_flat_video` with
//! `video_num_patches`,
//! `frames_indices` and `frame_duration_ms`. Their tubelets hold
//! `video_temporal_patch_size` frames, and the engine embeds a video's whole
//! replacement (it embeds the separators itself), so the whole replacement
//! is the feature range.

use std::collections::HashMap;

use serde_json::Value;

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
    },
    vision::{
        processors::nemotron_omni::{
            grid_tokens, NemotronConfigError, NemotronGeometry, DYNAMIC_IMAGE_SIDE_TENSORS,
            VIDEO_TEMPORAL_PATCH_SIZE,
        },
        NemotronOmniProcessor, PreProcessorConfig, VisionPreProcessor,
    },
};

/// Nemotron 3 Nano Omni, the checkpoint the `Packed` instance serves.
const MODEL_TYPE: &str = "NemotronH_Nano_Omni_Reasoning_V3";
/// The engine's architectures for this processor.
const ARCHITECTURES: [&str; 4] = [
    "NemotronH_Nano_VL_V2",
    MODEL_TYPE,
    "NemotronH_Super_Omni_Reasoning_V3",
    "NemotronH_Omni_Reasoning_V3",
];
const IMAGE_PLACEHOLDER: &str = "<image>";
const AUDIO_PLACEHOLDER: &str = "<so_embedding>";
/// What the chat template writes for a video, and the reserved token the
/// gateway rewrites it to as the video's anchor.
const VIDEO_MARKER: &str = "<video>";
const VIDEO_ANCHOR: &str = "<SPECIAL_30>";
/// The engine's default video loader keeps at most this many frames of a clip.
const MAX_SAMPLED_FRAMES: usize = 32;
/// A Reference video's tensors beside its frames, one entry per video, as
/// the engine declares them. `frames_indices` is as long as its clip, so
/// clips of different lengths stay separate batches.
const VIDEO_SIDE_TENSORS: [&str; 3] = ["video_num_patches", "frames_indices", "frame_duration_ms"];

/// The family's spec, one instance per tensor layout (see the module docs).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum NemotronOmniSpec {
    /// Nemotron 3 Nano Omni as packed patch rows, with audio.
    Packed,
    /// The engine's tensors for checkpoints whose images are sized dynamically.
    Reference,
    /// The engine's tensors for checkpoints whose images are tiled.
    ReferenceTiled,
}

impl NemotronOmniSpec {
    fn unsupported(self, modality: Modality) -> ModelRegistryError {
        ModelRegistryError::unsupported(self.name(), modality)
    }

    /// The anchor token, the feature token id and the wrapping begin/end ids.
    /// Video frames fill `<image>` positions.
    fn tokens(
        self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<(&'static str, TokenId, TokenId, TokenId)> {
        let (anchor, feature, begin, end) = match modality {
            Modality::Image => (IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER, "<img>", "</img>"),
            Modality::Video => (VIDEO_ANCHOR, IMAGE_PLACEHOLDER, "<img>", "</img>"),
            Modality::Audio if self == Self::Packed => (
                AUDIO_PLACEHOLDER,
                AUDIO_PLACEHOLDER,
                "<so_start>",
                "<so_end>",
            ),
            Modality::Audio | Modality::ImageEmbeds => return Err(self.unsupported(modality)),
        };
        Ok((
            anchor,
            metadata.token_id(feature)?,
            metadata.token_id(begin)?,
            metadata.token_id(end)?,
        ))
    }

    /// The checkpoint's geometry, as the Reference processor reads it.
    fn geometry(config: &Value) -> RegistryResult<NemotronGeometry> {
        NemotronGeometry::from_model_config(config).map_err(Self::geometry_error)
    }

    fn geometry_error(error: NemotronConfigError) -> ModelRegistryError {
        let name = |ptr: &str| ptr.trim_start_matches('/').replace('/', ".");
        match error {
            NemotronConfigError::Missing(ptr) => {
                ModelRegistryError::MissingConfigField { field: name(ptr) }
            }
            NemotronConfigError::Invalid(ptr) => ModelRegistryError::InvalidConfigField {
                field: name(ptr),
                reason: "the engine's processor refuses this value".to_string(),
            },
        }
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

    /// `(frames, rows, cols)` per video, as the Packed processor emits them.
    fn video_grids(preprocessed: &PreprocessedEncoderInputs) -> RegistryResult<Vec<[usize; 3]>> {
        let invalid = || ModelRegistryError::InvalidPreprocessedField {
            field: "video_grid_thw".to_string(),
        };
        let data = Self::int_values(preprocessed, "video_grid_thw")?;
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

    /// The separator before tubelet `tubelet` of `size` frames: its frames
    /// and their timestamps (`seconds`, one per frame of the clip, printed as
    /// Python's `:.2f` prints them).
    fn tubelet_separator(tubelet: usize, size: usize, seconds: &[f64]) -> String {
        let first = tubelet * size;
        let parts: Vec<String> = (first..(first + size).min(seconds.len()))
            .map(|frame| {
                let word = if frame == first { "Frame" } else { "frame" };
                format!(
                    "{word} {} sampled at {:.2} seconds",
                    frame + 1,
                    seconds[frame]
                )
            })
            .collect();
        let separator = format!("{}: ", parts.join(" and "));
        if tubelet > 0 {
            format!("\n{separator}")
        } else {
            separator
        }
    }

    /// Each video's stamped tubelets. A frame's timestamp is its source index
    /// times `int(1000 / fps)` milliseconds: Packed takes the indices and the
    /// rate from the decoder's report in `media`, Reference from the tensors
    /// its processor emits.
    fn video_replacements(
        self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        media: &[MediaItemInfo],
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let (anchor, token, begin, end) = self.tokens(metadata, Modality::Video)?;
        let invalid = |field: &str| ModelRegistryError::InvalidPreprocessedField {
            field: field.to_string(),
        };
        // One video, from its frames per tubelet, its tokens per tubelet and
        // its frames' timestamps.
        let replacement = |size: usize, per_tubelet: usize, seconds: Vec<f64>| {
            let mut tokens = Vec::new();
            let mut ranges = Vec::new();
            for tubelet in 0..seconds.len().div_ceil(size) {
                let separator = Self::tubelet_separator(tubelet, size, &seconds);
                tokens.extend(metadata.encode_plain(self.name(), &separator)?);
                tokens.push(begin);
                ranges.push(PlaceholderRange {
                    offset: tokens.len(),
                    length: per_tubelet,
                });
                tokens.extend(std::iter::repeat_n(token, per_tubelet));
                tokens.push(end);
            }
            if self != Self::Packed {
                // The engine embeds the whole replacement.
                ranges = vec![PlaceholderRange {
                    offset: 0,
                    length: tokens.len(),
                }];
            }
            Ok(PromptReplacement::sequence(Modality::Video, anchor, tokens)
                .with_feature_ranges(ranges))
        };
        if self == Self::Packed {
            return Self::video_grids(preprocessed)?
                .into_iter()
                .enumerate()
                .map(|(item, [frames, rows, cols])| {
                    let sampling = media
                        .get(item)
                        .and_then(|m| m.video_sampling.as_ref())
                        .filter(|s| s.source_fps.is_finite() && s.source_fps > 0.0)
                        .ok_or_else(|| {
                            invalid("video frame times (the decoder reported no frame rate)")
                        })?;
                    let frame_ms = (1000.0 / sampling.source_fps).trunc();
                    let indices = sampling
                        .frame_indices
                        .get(..frames)
                        .ok_or_else(|| invalid("video frame_indices"))?;
                    replacement(
                        VIDEO_TEMPORAL_PATCH_SIZE,
                        grid_tokens(rows, cols),
                        indices
                            .iter()
                            .map(|&index| index as f64 * frame_ms / 1000.0)
                            .collect(),
                    )
                })
                .collect();
        }
        let geometry = Self::geometry(metadata.config)?;
        let frames_per_video = Self::int_values(preprocessed, "video_num_patches")?;
        let durations = Self::int_values(preprocessed, "frame_duration_ms")?;
        let mut indices = Self::int_values(preprocessed, "frames_indices")?;
        if frames_per_video.len() != preprocessed.feature_token_counts.len()
            || durations.len() != frames_per_video.len()
        {
            return Err(invalid("video_num_patches"));
        }
        frames_per_video
            .iter()
            .zip(durations)
            .zip(&preprocessed.feature_token_counts)
            .map(|((&frames, &frame_ms), &count)| {
                let frames = usize::try_from(frames)
                    .ok()
                    .filter(|&f| f > 0 && f <= indices.len())
                    .ok_or_else(|| invalid("frames_indices"))?;
                let (own, rest) = indices.split_at(frames);
                indices = rest;
                replacement(
                    geometry.video_temporal_patch_size.max(1),
                    count / geometry.tubelets(frames),
                    own.iter()
                        .map(|&index| index as f64 * frame_ms as f64 / 1000.0)
                        .collect(),
                )
            })
            .collect()
    }

    /// A Reference modality's tensors: the engine's name for its pixels, its
    /// side tensors (one entry per item, kept on the CPU), and the one of
    /// them its pixels are cut by when they are not batched.
    fn reference_tensors(
        self,
        modality: Modality,
    ) -> Option<(&'static str, &'static [&'static str], Option<&'static str>)> {
        Some(match (self, modality) {
            (Self::Reference, Modality::Image) => (
                "pixel_values_flat",
                &DYNAMIC_IMAGE_SIDE_TENSORS,
                Some("image_num_patches"),
            ),
            (Self::ReferenceTiled, Modality::Image) => (
                "pixel_values_flat",
                &["image_num_patches"],
                Some("image_num_patches"),
            ),
            (Self::Reference | Self::ReferenceTiled, Modality::Video) => (
                "pixel_values_flat_video",
                &VIDEO_SIDE_TENSORS,
                Some("video_num_patches"),
            ),
            _ => return None,
        })
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
        ParakeetAudioParams::from_model_config(config)
            .map(|_| ())
            .map_err(Self::audio_config_error)
    }

    /// The registry's reading of a `sound_config` the audio port refused:
    /// a field it needs is missing, or set to a value it does not implement.
    fn audio_config_error(error: ParakeetConfigError) -> ModelRegistryError {
        match error {
            ParakeetConfigError::Missing(field) => ModelRegistryError::MissingConfigField {
                field: format!("sound_config.{field}"),
            },
            ParakeetConfigError::Unsupported(field) => {
                ModelRegistryError::UnsupportedModel(format!(
                    "Nemotron-H Omni config sound_config.{field} differs from the gateway's \
                     preprocessing"
                ))
            }
        }
    }
}

impl ModelProcessorSpec for NemotronOmniSpec {
    fn name(&self) -> &'static str {
        match self {
            Self::Packed => "nemotron_omni",
            Self::Reference => "nemotron_vl",
            Self::ReferenceTiled => "nemotron_vl_tiled",
        }
    }

    /// Packed serves its one checkpoint; a Reference instance serves the
    /// family's architectures whose images are laid out its way.
    fn matches(&self, metadata: &ModelMetadata) -> bool {
        if *self == Self::Packed {
            return metadata.config_model_type() == Some(MODEL_TYPE);
        }
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
                == (*self == Self::Reference)
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
        Ok(self.tokens(metadata, modality)?.0.to_string())
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        Ok(self.tokens(metadata, modality)?.1)
    }

    /// The Packed processors emit packed patch rows and mel frames, not the
    /// engine's `pixel_values_flat` / `pixel_values_flat_video` inputs.
    fn encoder_input_format(&self) -> EncoderInputFormat {
        match self {
            Self::Packed => EncoderInputFormat::Packed,
            Self::Reference | Self::ReferenceTiled => EncoderInputFormat::Reference,
        }
    }

    fn rendered_marker_for(&self, modality: Modality) -> Option<&'static str> {
        (modality == Modality::Video).then_some(VIDEO_MARKER)
    }

    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::UpTo {
            max_frames: MAX_SAMPLED_FRAMES,
        }
    }

    /// The family sets no per-request media count of its own for any of its
    /// three modalities: its serving engines read all three as unlimited (the
    /// family's processing info declares `image: None`, and `video: None` and
    /// `audio: None` for a checkpoint that carries them), the checkpoint's
    /// config bounds an image's patches and a clip's frames and parameterizes
    /// the sound encoder, never how many items of each, and the context
    /// length is the practical bound. The modalities the checkpoint carries
    /// stay declared (an undeclared modality is refused as unsupported)
    /// without a bound of the spec's, so the effective limit per modality is
    /// the engine's advertised one, else the deployment's
    /// `--mm-per-request-image-limit` / `SMG_IMAGE_MAX_COUNT`,
    /// `SMG_VIDEO_MAX_COUNT` or `SMG_AUDIO_MAX_COUNT`.
    ///
    /// What still bounds such a request, and where it lives. Before any
    /// fetch: the per-request limit the model's workers advertise for their
    /// engine, which caps the router's own limit in the gateway's limit
    /// resolution, and the deployment's `SMG_IMAGE_MAX_COUNT` /
    /// `SMG_VIDEO_MAX_COUNT` / `SMG_AUDIO_MAX_COUNT` (and
    /// `--mm-per-request-image-limit` for images), read in the same
    /// resolution. At or after the fetch: the gateway's cap on media bytes
    /// in flight (`multimodal_max_inflight_bytes`), the engine's advertised
    /// encoder budget and the context length. A number of the spec's own
    /// would be a declared default again, not a bound of the family's.
    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        let mut limits =
            HashMap::from([(Modality::Image, usize::MAX), (Modality::Video, usize::MAX)]);
        if *self != Self::Packed {
            // The Reference instances serve no audio.
            Self::geometry(metadata.config)?;
            return Ok(limits);
        }
        Self::check_config(metadata.config)?;
        if Self::has_audio(metadata.config) {
            limits.insert(Modality::Audio, usize::MAX);
        }
        Ok(limits)
    }

    fn vision_processor(
        &self,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image | Modality::Video => Ok(Box::new(if *self == Self::Packed {
                NemotronOmniProcessor::new()
            } else {
                // The processor config wins where it names a value: Nemotron
                // 3.5 Super VL sizes its video frames only there.
                let geometry = Self::geometry(model_config)?
                    .with_preprocessor_config(preprocessor_config)
                    .map_err(|error| ModelRegistryError::InvalidPreprocessorConfig {
                        spec: self.name(),
                        modality,
                        reason: match error {
                            NemotronConfigError::Missing(key) => format!("{key} is missing"),
                            NemotronConfigError::Invalid(key) => {
                                format!("{key} holds a value the engine's processor refuses")
                            }
                        },
                    })?;
                NemotronOmniProcessor::reference(geometry)
            })),
            _ => Err(ModelRegistryError::unsupported(self.name(), modality)),
        }
    }

    fn audio_processor(
        &self,
        model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Option<Box<dyn AudioPreProcessor>>> {
        // No `sound_config` is a model without audio; a `sound_config` the
        // port cannot serve is the configuration error `check_config` reports,
        // not a missing processor. The Reference instances serve no audio.
        if *self != Self::Packed || !Self::has_audio(model_config) {
            return Ok(None);
        }
        let params = ParakeetAudioParams::from_model_config(model_config)
            .map_err(Self::audio_config_error)?;
        Ok(Some(Box::new(ParakeetAudioProcessor::new(params))))
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
            Modality::Video => self.video_replacements(metadata, preprocessed, media),
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
            return self.video_replacements(metadata, preprocessed, &[]);
        }
        let (placeholder, token, begin, end) = self.tokens(metadata, modality)?;
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
        if *self != Self::Packed {
            let (_, side, sizes) = self.reference_tensors(modality).unwrap_or_default();
            return EncoderFieldLayouts::new(
                sizes.map_or(FieldLayout::Batched, FieldLayout::flat),
                side.iter()
                    .map(|key| ((*key).to_string(), FieldLayout::Batched))
                    .collect(),
            );
        }
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

    fn keep_on_cpu_keys_for(&self, modality: Modality) -> Vec<String> {
        let (_, side, _) = self.reference_tensors(modality).unwrap_or_default();
        side.iter().map(|key| (*key).to_string()).collect()
    }

    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        self.reference_tensors(modality)
            .map(|(key, ..)| key.to_string())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::{
            test_helpers::{test_preprocessed_with_tokens, TestTokenizer},
            ModelRegistry,
        },
        types::{ImageSize, VideoSamplingInfo},
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
    fn audio_processor_tells_no_audio_from_an_audio_config_it_cannot_serve() {
        let spec = NemotronOmniSpec::Packed;
        let default = PreProcessorConfig::default();

        let mut config = checkpoint_config();
        assert!(spec.audio_processor(&config, &default).unwrap().is_some());

        config["sound_config"]["subsampling_factor"] = json!(4);
        assert!(matches!(
            spec.audio_processor(&config, &default).err().unwrap(),
            ModelRegistryError::UnsupportedModel(message)
                if message.contains("sound_config.subsampling_factor")
        ));

        config["sound_config"] = json!({"sampling_rate": 16000});
        assert!(matches!(
            spec.audio_processor(&config, &default).err().unwrap(),
            ModelRegistryError::MissingConfigField { field }
                if field.starts_with("sound_config.")
        ));

        config.as_object_mut().unwrap().remove("sound_config");
        assert!(spec.audio_processor(&config, &default).unwrap().is_none());
    }

    #[test]
    fn refuses_a_config_off_the_implemented_geometry() {
        let tokenizer = tokenizer();
        let spec = NemotronOmniSpec::Packed;
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

    /// The family bounds its media by context, not by count: the spec
    /// declares its three modalities without a limit of its own (the 16
    /// images, 2 clips and 4 recordings it carried were declared defaults,
    /// never counts of the family's), so a deployment knob or the engine's
    /// advertised limit is the only router-side bound for each.
    #[test]
    fn nemotron_omni_declares_its_media_without_a_count_of_its_own() {
        let tokenizer = tokenizer();
        let config = checkpoint_config();
        let metadata = ModelMetadata {
            model_id: "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = NemotronOmniSpec::Packed;

        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([
                (Modality::Image, usize::MAX),
                (Modality::Video, usize::MAX),
                (Modality::Audio, usize::MAX),
            ])
        );
        for (modality, cap) in [
            (Modality::Image, 16),
            (Modality::Video, 2),
            (Modality::Audio, 4),
        ] {
            for count in [cap + 1, 1_000] {
                spec.validate_media_request(&metadata, &[(modality, count)])
                    .unwrap_or_else(|error| panic!("{count} {modality} items: {error}"));
            }
            // A deployment's own cap (the flag or the environment override, or
            // the engine's advertised limit the gateway passes in) still binds.
            let capped = HashMap::from([(modality, cap)]);
            assert_eq!(
                spec.validate_media_request_with_limits(&metadata, &[(modality, cap + 1)], &capped),
                Err(ModelRegistryError::ModalityLimitExceeded {
                    spec: "nemotron_omni",
                    modality,
                    limit: cap,
                    requested: cap + 1,
                })
            );
        }
    }

    #[test]
    fn preprocesses_packed_inputs_with_the_reference_loader_sampling() {
        let spec = NemotronOmniSpec::Packed;
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
        let spec = NemotronOmniSpec::Packed;
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
        let spec = NemotronOmniSpec::Packed;
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

    /// [`checkpoint_config`] with the fields only the Reference layout reads.
    fn omni_config() -> Value {
        let mut config = checkpoint_config();
        config["architectures"] = json!([MODEL_TYPE]);
        config["force_image_size"] = json!(512);
        config
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
        let tiled = NemotronOmniSpec::ReferenceTiled;
        let dynamic = NemotronOmniSpec::Reference;
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

    /// The processor config's fault is reported as its own, not as the
    /// model config's.
    #[test]
    fn reports_a_processor_config_the_engine_refuses() {
        let processor_config = PreProcessorConfig::from_value(json!({"max_num_tiles": 0})).unwrap();
        assert!(matches!(
            NemotronOmniSpec::ReferenceTiled.vision_processor(
                &v2_config(),
                &processor_config,
                Modality::Image
            ),
            Err(ModelRegistryError::InvalidPreprocessorConfig {
                spec: "nemotron_vl_tiled",
                modality: Modality::Image,
                ..
            })
        ));
    }

    #[test]
    fn names_the_engines_tensors() {
        let tiled = NemotronOmniSpec::ReferenceTiled;
        let dynamic = NemotronOmniSpec::Reference;
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
            FieldLayout::flat("image_num_patches")
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
        let spec = NemotronOmniSpec::Reference;
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
    fn stamps_each_reference_video_tubelet_with_its_frames() {
        const TEXT_BASE: TokenId = 1000;
        let tokenizer = tokenizer().with_byte_encoder(TEXT_BASE.unsigned_abs());
        let text = |s: &str| {
            s.bytes()
                .map(|b| TEXT_BASE + TokenId::from(b))
                .collect::<Vec<_>>()
        };
        let spec = NemotronOmniSpec::Reference;
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
