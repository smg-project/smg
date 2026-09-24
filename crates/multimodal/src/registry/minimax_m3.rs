use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    media::FrameSampling,
    registry::{
        MediaItemInfo, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{
        EncoderFieldLayouts, FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId,
        VideoSamplingInfo,
    },
    vision::{MiniMaxM3VisionProcessor, PreProcessorConfig},
};

/// Maximum images accepted in one request (MiniMax-M3 spec 1.3.6).
const MAX_IMAGES_PER_REQUEST: usize = 200;

/// Maximum videos accepted in one request (MiniMax-M3 spec 1.3.6).
const MAX_VIDEOS_PER_REQUEST: usize = 20;

/// Frame rate the reference `video_processor.py` samples at.
const DEFAULT_VIDEO_SAMPLE_FPS: f32 = 1.0;

/// MiniMax-M3 vision spec.
///
/// M3's media tokens carry the same `]<]...[>[` namespace framing as its tool
/// calls. Unlike the Qwen templates, M3's chat template renders a bare
/// `]<]image[>[` (or `]<]video[>[`) with no surrounding markers, so this spec
/// owns the whole wrapper: each placeholder expands to
/// `<start> + N * <pad> + <end>`, with
/// N = `grid_t * grid_h * grid_w / merge_size^2`. That mirrors vLLM's
/// `_get_prompt_updates`, which builds
/// `[start_token_id] + [image_token_id] * N + [end_token_id]`.
///
/// Both modalities are wrapped with the image markers: the vocabulary also
/// carries a `]<]start of video[>[` / `]<]end of video[>[` pair, but the
/// reference processor never emits it, so neither does this spec.
pub(super) struct MiniMaxM3VisionSpec;

/// What stamps one clip's frames: its sampling and the frames per temporal patch it was preprocessed with.
#[derive(Clone, Copy)]
struct ClipStamps<'a> {
    sampling: &'a VideoSamplingInfo,
    temporal_patch_size: usize,
}

impl ClipStamps<'_> {
    /// One stamp per temporal frame, or `None` when the clip cannot be stamped.
    fn texts(&self, grid_t: usize) -> Option<Vec<String>> {
        (0..grid_t)
            .map(|frame| {
                MiniMaxM3VisionSpec::timestamp_text(frame, self.temporal_patch_size, self.sampling)
            })
            .collect()
    }
}

impl MiniMaxM3VisionSpec {
    const IMAGE_TOKEN: &'static str = "]<]image[>[";
    const VIDEO_TOKEN: &'static str = "]<]video[>[";
    const IMAGE_START_TOKEN: &'static str = "]<]start of image[>[";
    const IMAGE_END_TOKEN: &'static str = "]<]end of image[>[";

    /// The structural markers wrapping one modality's feature run.
    fn wrapper_tokens(modality: Modality) -> RegistryResult<(&'static str, &'static str)> {
        match modality {
            Modality::Image | Modality::Video => {
                Ok((Self::IMAGE_START_TOKEN, Self::IMAGE_END_TOKEN))
            }
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: "minimax_m3",
                modality,
            }),
        }
    }

    /// The repeated feature token for images.
    ///
    /// `image_token_index` is the checkpoint's own declaration; the tokenizer
    /// lookup is the fallback for checkpoints that omit it.
    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        match metadata.config_u32(&["image_token_index"]) {
            Some(id) => Ok(id as TokenId),
            None => metadata.token_id(Self::IMAGE_TOKEN),
        }
    }

    /// The repeated feature token for videos.
    fn video_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        match metadata.config_u32(&["video_token_index"]) {
            Some(id) => Ok(id as TokenId),
            None => metadata.token_id(Self::VIDEO_TOKEN),
        }
    }

    /// Whether the checkpoint declares video support.
    fn supports_video(metadata: &ModelMetadata) -> bool {
        metadata.config_u32(&["video_token_index"]).is_some()
            || metadata.token_id(Self::VIDEO_TOKEN).is_ok()
    }

    /// Encode timestamp text on its own with no special tokens, as vLLM does.
    fn encode_plain_text(metadata: &ModelMetadata, text: &str) -> RegistryResult<Vec<TokenId>> {
        let ids = metadata.tokenizer.encode_text(text).ok_or_else(|| {
            ModelRegistryError::TextEncodingFailed {
                spec: "minimax_m3",
                text: text.to_string(),
            }
        })?;
        Ok(ids.into_iter().map(|id| id as TokenId).collect())
    }

    /// The `]<]X.X seconds[>[` stamp for one temporal frame, from its first sampled source frame clamped to the last one; `None` when nothing was sampled.
    fn timestamp_text(
        frame: usize,
        temporal_patch_size: usize,
        sampling: &VideoSamplingInfo,
    ) -> Option<String> {
        let last = sampling.frame_indices.len().checked_sub(1)?;
        let source_index = sampling.frame_indices[(frame * temporal_patch_size).min(last)];
        let seconds = source_index as f64 / sampling.source_fps;
        Some(format!("]<]{seconds:.1} seconds[>["))
    }

    /// A clip's sampling, when the decoder reported a usable source fps.
    fn video_sampling(item: &MediaItemInfo) -> Option<&VideoSamplingInfo> {
        item.video_sampling
            .as_ref()
            .filter(|sampling| sampling.source_fps.is_finite() && sampling.source_fps > 0.0)
    }

    /// Build `[start] + N * pad + [end]` for one media item.
    fn wrapped_replacement(
        metadata: &ModelMetadata,
        modality: Modality,
        placeholder_token: &str,
        pad_token_id: TokenId,
        num_tokens: usize,
    ) -> RegistryResult<PromptReplacement> {
        let (start_token, end_token) = Self::wrapper_tokens(modality)?;
        let start_id = metadata.token_id(start_token)?;
        let end_id = metadata.token_id(end_token)?;

        let mut tokens = Vec::with_capacity(num_tokens + 2);
        tokens.push(start_id);
        tokens.extend(std::iter::repeat_n(pad_token_id, num_tokens));
        tokens.push(end_id);

        Ok(
            PromptReplacement::sequence(modality, placeholder_token, tokens)
                // The encoder features occupy only the padded middle; the two
                // markers around them are structural.
                //
                // `structural_prefix` stays 0: it counts markers the chat
                // template emits *before* the placeholder, which `expand_tokens`
                // folds in by widening the range backwards without re-emitting
                // them. M3's template emits a bare placeholder and both markers
                // are inside `tokens`, so a non-zero prefix would report a range
                // starting one token too early.
                .with_feature_span(1, num_tokens),
        )
    }

    /// Temporal grid depth of each video, one per row of `video_grid_thw`.
    ///
    /// The video path always emits this tensor and the per-frame layout is
    /// built from it, so an absent or malformed grid is a broken preprocessing
    /// output. It is rejected rather than papered over with a flat block the
    /// model would read differently.
    fn video_grid_ts(preprocessed: &PreprocessedEncoderInputs) -> RegistryResult<Vec<usize>> {
        let invalid = || ModelRegistryError::InvalidPreprocessedField {
            field: "video_grid_thw".to_string(),
        };
        match preprocessed.model_specific.get("video_grid_thw") {
            Some(ModelSpecificValue::IntTensor { data, shape })
                if shape.len() == 2
                    && shape[1] == 3
                    && shape[0] > 0
                    && data.len() == shape[0] * 3 =>
            {
                data.chunks(3)
                    .map(|row| usize::try_from(row[0]).map_err(|_| invalid()))
                    .collect()
            }
            _ => Err(invalid()),
        }
    }

    /// Build the per-frame video body and the feature range of each frame.
    ///
    /// M3 lays video out as one `]<]start of image[>[` .. `]<]end of image[>[`
    /// block **per temporal frame**, each holding `grid_h * grid_w / merge^2`
    /// pad tokens — not one flat block over the whole clip, and with the image
    /// markers rather than the vocabulary's unused video pair. When the clip's
    /// stamps are known, a `]<]X.X seconds[>[` stamp precedes each block. The
    /// reference processor builds the same shape:
    ///
    /// ```text
    /// for frame in 0..grid_t:
    ///     [X.X seconds if sampled] + [start of image] + [video_token] * M + [end of image]
    /// ```
    ///
    /// Ranges are running offsets because stamps tokenize to different lengths; only pad runs are features.
    /// `None` for an unstamped single-frame clip (one block is the same layout); a token count that does not divide by the frame count is rejected.
    fn per_frame_video_tokens(
        metadata: &ModelMetadata,
        pad_token_id: TokenId,
        num_tokens: usize,
        grid_t: usize,
        stamps: Option<ClipStamps<'_>>,
    ) -> RegistryResult<Option<(Vec<TokenId>, Vec<PlaceholderRange>)>> {
        let stamps = stamps.and_then(|clip| clip.texts(grid_t));
        if num_tokens == 0 || (grid_t <= 1 && stamps.is_none()) {
            return Ok(None);
        }
        if !num_tokens.is_multiple_of(grid_t) {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw".to_string(),
            });
        }
        let start_id = metadata.token_id(Self::IMAGE_START_TOKEN)?;
        let end_id = metadata.token_id(Self::IMAGE_END_TOKEN)?;

        let per_frame = num_tokens / grid_t;
        let mut tokens = Vec::with_capacity(num_tokens + 2 * grid_t);
        let mut ranges = Vec::with_capacity(grid_t);
        for frame in 0..grid_t {
            if let Some(stamps) = &stamps {
                tokens.extend(Self::encode_plain_text(metadata, &stamps[frame])?);
            }
            tokens.push(start_id);
            ranges.push(PlaceholderRange {
                offset: tokens.len(),
                length: per_frame,
            });
            tokens.extend(std::iter::repeat_n(pad_token_id, per_frame));
            tokens.push(end_id);
        }
        Ok(Some((tokens, ranges)))
    }

    /// One replacement per clip, stamped where `stamps` carries that clip's sampling.
    fn video_replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        placeholder_token: &str,
        stamps: &[Option<ClipStamps<'_>>],
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::video_token_id(metadata)?;
        let grid_ts = Self::video_grid_ts(preprocessed)?;
        if grid_ts.len() != preprocessed.feature_token_counts.len() {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw".to_string(),
            });
        }

        preprocessed
            .feature_token_counts
            .iter()
            .zip(grid_ts)
            .enumerate()
            .map(|(index, (&num_tokens, grid_t))| {
                let per_frame = Self::per_frame_video_tokens(
                    metadata,
                    pad_token_id,
                    num_tokens,
                    grid_t,
                    stamps.get(index).copied().flatten(),
                )?;

                match per_frame {
                    Some((tokens, ranges)) => {
                        Ok(
                            PromptReplacement::sequence(Modality::Video, placeholder_token, tokens)
                                .with_feature_ranges(ranges),
                        )
                    }
                    None => Self::wrapped_replacement(
                        metadata,
                        Modality::Video,
                        placeholder_token,
                        pad_token_id,
                        num_tokens,
                    ),
                }
            })
            .collect()
    }

    fn replacements_for(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
        placeholder_token: &str,
        pad_token_id: TokenId,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                Self::wrapped_replacement(
                    metadata,
                    modality,
                    placeholder_token,
                    pad_token_id,
                    num_tokens,
                )
            })
            .collect()
    }
}

impl ModelProcessorSpec for MiniMaxM3VisionSpec {
    fn name(&self) -> &'static str {
        "minimax_m3"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        if metadata
            .config_model_type()
            .is_some_and(|mt| mt == "minimax_m3_vl")
        {
            return true;
        }
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("minimax") && id.contains("m3")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(Self::IMAGE_TOKEN.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_token_id(metadata)
    }

    /// The bare `]<]image[>[` / `]<]video[>[` are vLLM's single-token targets.
    fn worker_expandable(&self, modality: Modality) -> bool {
        matches!(modality, Modality::Image | Modality::Video)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            Modality::Video => Ok(Self::VIDEO_TOKEN.to_string()),
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
            Modality::Image => Self::image_token_id(metadata),
            Modality::Video => Self::video_token_id(metadata),
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
        // MiniMax-M3 accepts up to 200 images per request (spec 1.3.6), far
        // above the Qwen-family default of 10.
        let mut limits = HashMap::from([(Modality::Image, MAX_IMAGES_PER_REQUEST)]);
        if Self::supports_video(metadata) {
            limits.insert(Modality::Video, MAX_VIDEOS_PER_REQUEST);
        }
        Ok(limits)
    }

    /// Matching the reference sampling rate keeps a default video request at
    /// the reference's frame and token counts.
    fn default_video_sample_fps(&self) -> Option<f32> {
        Some(DEFAULT_VIDEO_SAMPLE_FPS)
    }

    /// The reference takes one frame per second from the start and always
    /// keeps the last frame.
    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::Interval
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::image_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        Self::replacements_for(
            metadata,
            preprocessed,
            Modality::Image,
            &placeholder_token,
            pad_token_id,
        )
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.prompt_replacements(metadata, preprocessed),
            Modality::Video => {
                let placeholder_token = self.placeholder_token_for(metadata, Modality::Video)?;
                Self::video_replacements(metadata, preprocessed, &placeholder_token, &[])
            }
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

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
                let placeholder_token = self.placeholder_token_for(metadata, Modality::Video)?;
                let temporal_patch_size =
                    MiniMaxM3VisionProcessor::temporal_patch_size_from(preprocessor_config);
                let stamps: Vec<Option<ClipStamps<'_>>> = media
                    .iter()
                    .map(|item| {
                        Self::video_sampling(item).map(|sampling| ClipStamps {
                            sampling,
                            temporal_patch_size,
                        })
                    })
                    .collect();
                Self::video_replacements(metadata, preprocessed, &placeholder_token, &stamps)
            }
            _ => self.prompt_replacements_for(metadata, preprocessed, modality),
        }
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Mirrors vLLM's `_get_mm_fields_config` for M3: the pixel tensors are
        // flat over patches and sliced per item by the grid product, while the
        // grid triples are batched one row per item. The frame spacing the
        // shared video processor emits is listed too: M3 reads the timing off
        // its text stamps and never looks at it, but a value with no layout is
        // one every clip has to agree on, and clips sampled at different rates
        // do not.
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
            (
                "pixel_values_videos".to_string(),
                FieldLayout::flat("patches_per_video"),
            ),
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_video".to_string(), FieldLayout::Batched),
            ("video_second_per_grid".to_string(), FieldLayout::Batched),
        ])
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        // One map per modality: a video batch must not advertise the image
        // sizes key (or the other way round), or a request carrying both is
        // sliced by a tensor the batch does not have.
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
        // vLLM marks both grid tensors keep_on_cpu=True.
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use serde::Deserialize;
    use serde_json::json;

    use super::*;
    use crate::registry::{test_helpers::TestTokenizer, ModelMetadata, Tokenizer};

    /// Vocabulary ids for M3's media markers, as the checkpoint declares them.
    const IMAGE_ID: TokenId = 200_025;
    const VIDEO_ID: TokenId = 200_026;
    const IMAGE_START_ID: TokenId = 200_029;
    const IMAGE_END_ID: TokenId = 200_030;
    // Present in the vocabulary but never emitted by the reference processor.
    const VIDEO_START_ID: TokenId = 200_031;
    const VIDEO_END_ID: TokenId = 200_032;
    /// Byte-encoder offset, chosen so text ids cannot collide with the marker ids.
    const TEXT_BASE: u32 = 1000;

    /// SHA-256 of the `processing_minimax.py` the timestamp fixture was recorded from.
    const REFERENCE_SHA256: &str =
        "0706ae8b93b3489a3b3bc7f48a5f29fb34b1449ebebb37586250e4d5fa8f5e35";

    fn m3_tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            (MiniMaxM3VisionSpec::IMAGE_TOKEN, IMAGE_ID as u32),
            (MiniMaxM3VisionSpec::VIDEO_TOKEN, VIDEO_ID as u32),
            (
                MiniMaxM3VisionSpec::IMAGE_START_TOKEN,
                IMAGE_START_ID as u32,
            ),
            (MiniMaxM3VisionSpec::IMAGE_END_TOKEN, IMAGE_END_ID as u32),
            ("]<]start of video[>[", VIDEO_START_ID as u32),
            ("]<]end of video[>[", VIDEO_END_ID as u32),
        ])
        .with_byte_encoder(TEXT_BASE)
    }

    fn tokenizer() -> &'static TestTokenizer {
        static TOKENIZER: OnceLock<TestTokenizer> = OnceLock::new();
        TOKENIZER.get_or_init(m3_tokenizer)
    }

    fn m3_config() -> Value {
        json!({
            "model_type": "minimax_m3_vl",
            "image_token_index": IMAGE_ID,
            "video_token_index": VIDEO_ID,
        })
    }

    fn metadata() -> ModelMetadata<'static> {
        static CONFIG: OnceLock<Value> = OnceLock::new();
        ModelMetadata {
            model_id: "MiniMaxAI/MiniMax-M3",
            config: CONFIG.get_or_init(m3_config),
            tokenizer: tokenizer(),
        }
    }

    /// Knows the markers but cannot encode plain text.
    struct NoTextTokenizer;

    impl Tokenizer for NoTextTokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            tokenizer().token_to_id(token)
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            tokenizer().id_to_token(id)
        }

        fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
            None
        }
    }

    fn text_ids(text: &str) -> Vec<TokenId> {
        text.bytes()
            .map(|b| (TEXT_BASE + u32::from(b)) as TokenId)
            .collect()
    }

    fn preprocessed(counts: Vec<usize>) -> PreprocessedEncoderInputs {
        let item_sizes = vec![(224, 224); counts.len()];
        PreprocessedEncoderInputs::new(ndarray::Array2::<f32>::zeros((1, 1)), counts, item_sizes)
    }

    #[test]
    fn matches_by_model_type_and_id() {
        let spec = MiniMaxM3VisionSpec;
        assert!(spec.matches(&metadata()));
    }

    #[test]
    fn placeholder_tokens_use_the_m3_namespace() {
        let spec = MiniMaxM3VisionSpec;
        let meta = metadata();

        assert_eq!(spec.placeholder_token(&meta).unwrap(), "]<]image[>[");
        assert_eq!(
            spec.placeholder_token_for(&meta, Modality::Video).unwrap(),
            "]<]video[>["
        );
        assert_eq!(spec.placeholder_token_id(&meta).unwrap(), IMAGE_ID);
        assert_eq!(
            spec.placeholder_token_id_for(&meta, Modality::Video)
                .unwrap(),
            VIDEO_ID
        );
    }

    #[test]
    fn image_replacement_is_wrapped_in_start_and_end_markers() {
        let spec = MiniMaxM3VisionSpec;
        let meta = metadata();
        let replacements = spec
            .prompt_replacements(&meta, &preprocessed(vec![4]))
            .unwrap();

        assert_eq!(replacements.len(), 1);
        let replacement = &replacements[0];

        // M3's chat template emits a bare ]<]image[>[, so the spec owns the
        // surrounding markers.
        assert_eq!(
            replacement.tokens,
            vec![
                IMAGE_START_ID,
                IMAGE_ID,
                IMAGE_ID,
                IMAGE_ID,
                IMAGE_ID,
                IMAGE_END_ID
            ]
        );
        assert_eq!(replacement.placeholder_token, "]<]image[>[");
        assert_eq!(replacement.modality, Modality::Image);
    }

    #[test]
    fn feature_span_skips_the_structural_markers() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements(&metadata(), &preprocessed(vec![4]))
            .unwrap();
        let ranges = replacements[0].feature_ranges.as_ref().unwrap();

        // The encoder features are the padded middle only.
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 4);
        // Both markers live inside `tokens`, so nothing is folded in from
        // before the placeholder.
        assert_eq!(replacements[0].structural_prefix, 0);
    }

    #[test]
    fn one_replacement_per_media_item() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements(&metadata(), &preprocessed(vec![2, 3]))
            .unwrap();

        assert_eq!(replacements.len(), 2);
        assert_eq!(replacements[0].tokens.len(), 2 + 2);
        assert_eq!(replacements[1].tokens.len(), 3 + 2);
    }

    #[test]
    fn video_replacement_uses_the_video_pad_token() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![3], 1),
                Modality::Video,
            )
            .unwrap();

        // Video pads sit inside the image markers; the vocabulary's video
        // marker pair is never emitted, as in the reference processor.
        assert_eq!(
            replacements[0].tokens,
            vec![IMAGE_START_ID, VIDEO_ID, VIDEO_ID, VIDEO_ID, IMAGE_END_ID]
        );
        assert!(!replacements[0].tokens.contains(&VIDEO_START_ID));
        assert!(!replacements[0].tokens.contains(&VIDEO_END_ID));
        assert_eq!(replacements[0].modality, Modality::Video);
        assert_eq!(replacements[0].placeholder_token, "]<]video[>[");
    }

    /// Preprocessed video carrying one `[grid_t, h, w]` row per clip.
    fn preprocessed_video(counts: Vec<usize>, grid_t: i64) -> PreprocessedEncoderInputs {
        let clips = counts.len();
        preprocessed(counts).with_extra(
            "video_grid_thw",
            ModelSpecificValue::IntTensor {
                data: [grid_t, 4, 4].repeat(clips),
                shape: vec![clips, 3],
            },
        )
    }

    #[test]
    fn multi_frame_video_emits_one_block_per_frame() {
        let spec = MiniMaxM3VisionSpec;
        // 3 temporal frames, 12 tokens total => 4 pad tokens per frame.
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![12], 3),
                Modality::Video,
            )
            .unwrap();

        let tokens = &replacements[0].tokens;
        // Each frame is [start of image] + 4 pads + [end of image], repeated
        // once per frame; the reference processor builds the same shape.
        let mut frame_tokens = vec![IMAGE_START_ID];
        frame_tokens.extend(std::iter::repeat_n(VIDEO_ID, 4));
        frame_tokens.push(IMAGE_END_ID);
        let expected: Vec<TokenId> = std::iter::repeat_n(frame_tokens, 3).flatten().collect();
        assert_eq!(tokens, &expected);
        assert_eq!(tokens.len(), 3 * (4 + 2));
    }

    #[test]
    fn multi_frame_feature_ranges_skip_each_frames_markers() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![12], 3),
                Modality::Video,
            )
            .unwrap();

        let ranges = replacements[0].feature_ranges.as_ref().unwrap();
        assert_eq!(ranges.len(), 3);
        // Frame f starts at f*(4+2), its pads begin one token later.
        assert_eq!((ranges[0].offset, ranges[0].length), (1, 4));
        assert_eq!((ranges[1].offset, ranges[1].length), (7, 4));
        assert_eq!((ranges[2].offset, ranges[2].length), (13, 4));
    }

    #[test]
    fn single_frame_video_stays_one_block() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![4], 1),
                Modality::Video,
            )
            .unwrap();

        assert_eq!(
            replacements[0].tokens,
            vec![
                IMAGE_START_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_ID,
                IMAGE_END_ID
            ]
        );
    }

    /// Media info for the request's one clip, with its sampling.
    fn sampled(frame_indices: &[usize], source_fps: f64) -> MediaItemInfo {
        MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps,
                frame_indices: frame_indices.to_vec(),
            }),
        }
    }

    /// One `[start of image] + pads + [end of image]` frame block.
    fn frame_block(pads: usize) -> Vec<TokenId> {
        let mut block = vec![IMAGE_START_ID];
        block.extend(std::iter::repeat_n(VIDEO_ID, pads));
        block.push(IMAGE_END_ID);
        block
    }

    fn range(offset: usize, length: usize) -> PlaceholderRange {
        PlaceholderRange { offset, length }
    }

    /// The video replacements for `counts` clips, given their media info and the preprocessor config.
    fn video_replacements_with(
        meta: &ModelMetadata,
        counts: Vec<usize>,
        grid_t: i64,
        media: &[MediaItemInfo],
        config: &PreProcessorConfig,
    ) -> Vec<PromptReplacement> {
        MiniMaxM3VisionSpec
            .prompt_replacements_with_media(
                meta,
                &preprocessed_video(counts, grid_t),
                Modality::Video,
                media,
                config,
            )
            .unwrap()
    }

    /// The video replacement for the request's one clip under the default preprocessor config.
    fn video_replacement(
        meta: &ModelMetadata,
        counts: Vec<usize>,
        grid_t: i64,
        media: &[MediaItemInfo],
    ) -> PromptReplacement {
        let mut replacements =
            video_replacements_with(meta, counts, grid_t, media, &PreProcessorConfig::default());
        assert_eq!(replacements.len(), 1);
        replacements.remove(0)
    }

    /// The stamp texts spliced into a replacement, decoded from their byte ids.
    fn stamps(replacement: &PromptReplacement) -> Vec<String> {
        let mut stamps = Vec::new();
        let mut text = Vec::new();
        for &id in &replacement.tokens {
            let byte = u32::try_from(id)
                .ok()
                .and_then(|id| id.checked_sub(TEXT_BASE))
                .and_then(|byte| u8::try_from(byte).ok());
            match byte {
                Some(byte) => text.push(byte),
                None if !text.is_empty() => {
                    stamps.push(String::from_utf8(std::mem::take(&mut text)).unwrap());
                }
                None => {}
            }
        }
        stamps
    }

    #[test]
    fn sampled_video_stamps_each_frame_with_its_source_time() {
        // Five frames at 30 fps pair into three temporal frames, each stamped with its first frame's time.
        let replacement = video_replacement(
            &metadata(),
            vec![12],
            3,
            &[sampled(&[0, 15, 30, 45, 60], 30.0)],
        );

        let mut expected = Vec::new();
        for stamp in [
            "]<]0.0 seconds[>[",
            "]<]1.0 seconds[>[",
            "]<]2.0 seconds[>[",
        ] {
            expected.extend(text_ids(stamp));
            expected.extend(frame_block(4));
        }
        assert_eq!(replacement.tokens, expected);
        // Each frame's pads follow its 17-byte stamp and start marker.
        assert_eq!(
            replacement.feature_ranges.as_ref().unwrap(),
            &vec![range(18, 4), range(41, 4), range(64, 4)]
        );
        // The stamps live inside `tokens`, so nothing is folded in from before the placeholder.
        assert_eq!(replacement.structural_prefix, 0);
        assert_eq!(replacement.modality, Modality::Video);
        assert_eq!(replacement.placeholder_token, "]<]video[>[");
    }

    #[test]
    fn feature_ranges_track_the_variable_width_stamps() {
        // A three-digit second count tokenizes longer, so ranges are running offsets, not a fixed stride.
        let replacement = video_replacement(
            &metadata(),
            vec![12],
            3,
            &[sampled(&[0, 1, 3000, 3001, 3002], 30.0)],
        );

        assert_eq!(
            stamps(&replacement),
            [
                "]<]0.0 seconds[>[",
                "]<]100.0 seconds[>[",
                "]<]100.1 seconds[>["
            ]
        );
        let ranges = replacement.feature_ranges.as_ref().unwrap();
        assert_eq!(ranges, &vec![range(18, 4), range(43, 4), range(68, 4)]);
        for range in ranges {
            let pads = &replacement.tokens[range.offset..range.offset + range.length];
            assert!(pads.iter().all(|&id| id == VIDEO_ID));
            assert_eq!(replacement.tokens[range.offset - 1], IMAGE_START_ID);
            assert_eq!(
                replacement.tokens[range.offset + range.length],
                IMAGE_END_ID
            );
        }
    }

    #[test]
    fn odd_frame_count_stamps_the_padded_pair_with_its_real_frame() {
        // Three frames pad to two temporal frames; the padded pair is stamped with its one real frame.
        let replacement =
            video_replacement(&metadata(), vec![8], 2, &[sampled(&[0, 15, 30], 30.0)]);

        assert_eq!(
            stamps(&replacement),
            ["]<]0.0 seconds[>[", "]<]1.0 seconds[>["]
        );
    }

    #[test]
    fn short_frame_index_list_clamps_to_the_last_sampled_frame() {
        // Frames past a short index list reuse the last index, as the reference's `min(.., len - 1)` does.
        let replacement = video_replacement(&metadata(), vec![12], 3, &[sampled(&[0, 30], 30.0)]);

        assert_eq!(
            stamps(&replacement),
            [
                "]<]0.0 seconds[>[",
                "]<]1.0 seconds[>[",
                "]<]1.0 seconds[>["
            ]
        );
    }

    #[test]
    fn single_frame_video_with_sampling_is_stamped() {
        // The reference loops over grid_t even when it is 1, so a sampled one-frame clip is stamped.
        let replacement = video_replacement(&metadata(), vec![4], 1, &[sampled(&[7], 25.0)]);

        let mut expected = text_ids("]<]0.3 seconds[>[");
        expected.extend(frame_block(4));
        assert_eq!(replacement.tokens, expected);
        assert_eq!(
            replacement.feature_ranges.as_ref().unwrap(),
            &vec![range(18, 4)]
        );
    }

    #[test]
    fn temporal_patch_size_from_the_preprocessor_config_picks_the_stamped_frames() {
        let meta = metadata();
        let indices = [0, 15, 30, 45, 60, 75, 90, 105];
        // The default pairs frames: four temporal frames, one second apart.
        let replacement = video_replacement(&meta, vec![16], 4, &[sampled(&indices, 30.0)]);
        assert_eq!(
            stamps(&replacement),
            [
                "]<]0.0 seconds[>[",
                "]<]1.0 seconds[>[",
                "]<]2.0 seconds[>[",
                "]<]3.0 seconds[>["
            ]
        );

        // A patch size of 4 makes the second stamp come from the fifth sampled frame, from either config spelling.
        let flat = PreProcessorConfig {
            temporal_patch_size: Some(4),
            ..Default::default()
        };
        let mut nested = PreProcessorConfig::default();
        nested.extra.insert(
            "img_token_compression_config".to_string(),
            json!({ "temporal_patch_size": 4 }),
        );
        for config in [flat, nested] {
            let replacements =
                video_replacements_with(&meta, vec![8], 2, &[sampled(&indices, 30.0)], &config);
            assert_eq!(
                stamps(&replacements[0]),
                ["]<]0.0 seconds[>[", "]<]2.0 seconds[>["]
            );
        }
    }

    #[test]
    fn each_clip_is_stamped_with_its_own_sampling() {
        let replacements = video_replacements_with(
            &metadata(),
            vec![12, 12],
            3,
            &[
                sampled(&[0, 15, 30, 45, 60], 30.0),
                sampled(&[300, 315, 330, 345, 360], 30.0),
            ],
            &PreProcessorConfig::default(),
        );

        assert_eq!(replacements.len(), 2);
        assert_eq!(
            stamps(&replacements[0]),
            [
                "]<]0.0 seconds[>[",
                "]<]1.0 seconds[>[",
                "]<]2.0 seconds[>["
            ]
        );
        assert_eq!(
            stamps(&replacements[1]),
            [
                "]<]10.0 seconds[>[",
                "]<]11.0 seconds[>[",
                "]<]12.0 seconds[>["
            ]
        );
    }

    #[test]
    fn clips_past_the_media_list_stay_unstamped() {
        let meta = metadata();
        let unstamped = MiniMaxM3VisionSpec
            .prompt_replacements_for(&meta, &preprocessed_video(vec![12], 3), Modality::Video)
            .unwrap();
        let replacements = video_replacements_with(
            &meta,
            vec![12, 12],
            3,
            &[sampled(&[0, 15, 30, 45, 60], 30.0)],
            &PreProcessorConfig::default(),
        );

        assert_eq!(stamps(&replacements[0]).len(), 3);
        assert_eq!(replacements[1].tokens, unstamped[0].tokens);
        assert_eq!(replacements[1].feature_ranges, unstamped[0].feature_ranges);
    }

    #[test]
    fn empty_frame_index_list_has_no_stamp() {
        let sampling = VideoSamplingInfo {
            source_fps: 30.0,
            frame_indices: Vec::new(),
        };
        assert_eq!(MiniMaxM3VisionSpec::timestamp_text(0, 2, &sampling), None);
    }

    #[test]
    fn unencodable_stamp_text_is_rejected() {
        let config = m3_config();
        let meta = ModelMetadata {
            model_id: "MiniMaxAI/MiniMax-M3",
            config: &config,
            tokenizer: &NoTextTokenizer,
        };
        let err = MiniMaxM3VisionSpec
            .prompt_replacements_with_media(
                &meta,
                &preprocessed_video(vec![12], 3),
                Modality::Video,
                &[sampled(&[0, 15, 30, 45, 60], 30.0)],
                &PreProcessorConfig::default(),
            )
            .unwrap_err();

        assert_eq!(
            err,
            ModelRegistryError::TextEncodingFailed {
                spec: "minimax_m3",
                text: "]<]0.0 seconds[>[".to_string(),
            }
        );
    }

    #[test]
    fn unusable_sampling_falls_back_to_the_unstamped_layout() {
        let meta = metadata();
        let unstamped = MiniMaxM3VisionSpec
            .prompt_replacements_for(&meta, &preprocessed_video(vec![12], 3), Modality::Video)
            .unwrap();

        for media in [
            vec![],
            vec![MediaItemInfo::default()],
            vec![sampled(&[], 30.0)],
            vec![sampled(&[0, 15, 30, 45, 60], 0.0)],
        ] {
            let replacement = video_replacement(&meta, vec![12], 3, &media);
            assert_eq!(replacement.tokens, unstamped[0].tokens);
            assert_eq!(replacement.feature_ranges, unstamped[0].feature_ranges);
        }
    }

    #[test]
    fn images_ignore_video_sampling() {
        let meta = metadata();
        let plain = MiniMaxM3VisionSpec
            .prompt_replacements(&meta, &preprocessed(vec![4]))
            .unwrap();
        let with_media = MiniMaxM3VisionSpec
            .prompt_replacements_with_media(
                &meta,
                &preprocessed(vec![4]),
                Modality::Image,
                &[sampled(&[0], 30.0)],
                &PreProcessorConfig::default(),
            )
            .unwrap();

        assert_eq!(with_media[0].tokens, plain[0].tokens);
        assert_eq!(with_media[0].feature_ranges, plain[0].feature_ranges);
    }

    #[derive(Deserialize)]
    struct TimestampGolden {
        reference: String,
        reference_sha256: String,
        generated_by: String,
        cases: Vec<TimestampCase>,
    }

    #[derive(Deserialize)]
    struct TimestampCase {
        name: String,
        source_fps: f64,
        frame_indices: Vec<usize>,
        temporal_patch_size: usize,
        grid_t: usize,
        timestamps: Vec<String>,
    }

    #[test]
    fn stamps_match_the_reference_processor() {
        let golden: TimestampGolden = serde_json::from_str(include_str!(
            "../../tests/fixtures/golden/minimax_m3_video_timestamps.json"
        ))
        .unwrap();
        assert_eq!(golden.reference, "processing_minimax.py");
        assert_eq!(golden.reference_sha256, REFERENCE_SHA256);
        assert!(golden.generated_by.contains(&golden.reference));
        assert!(!golden.cases.is_empty());

        for case in &golden.cases {
            let sampling = VideoSamplingInfo {
                source_fps: case.source_fps,
                frame_indices: case.frame_indices.clone(),
            };
            let stamps: Vec<String> = (0..case.grid_t)
                .map(|frame| {
                    MiniMaxM3VisionSpec::timestamp_text(frame, case.temporal_patch_size, &sampling)
                        .unwrap()
                })
                .collect();
            assert_eq!(stamps, case.timestamps, "case {}", case.name);
        }
    }

    #[test]
    fn ragged_token_count_is_rejected() {
        let spec = MiniMaxM3VisionSpec;
        // 10 tokens over 3 frames does not divide evenly, so the grid and the
        // token count disagree. Falling back to one flat block would silently
        // hand the model the framing the per-frame layout exists to avoid.
        let err = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![10], 3),
                Modality::Video,
            )
            .unwrap_err();

        assert!(matches!(
            err,
            ModelRegistryError::InvalidPreprocessedField { ref field } if field == "video_grid_thw"
        ));
    }

    #[test]
    fn missing_video_grid_is_rejected() {
        let spec = MiniMaxM3VisionSpec;
        // Without `video_grid_thw` the frame count is unknown, so the layout
        // cannot be built and a flat block would be wrong for any multi-frame
        // clip. The video path always emits the grid; its absence is a bug.
        let err = spec
            .prompt_replacements_for(&metadata(), &preprocessed(vec![12]), Modality::Video)
            .unwrap_err();

        assert!(matches!(
            err,
            ModelRegistryError::InvalidPreprocessedField { ref field } if field == "video_grid_thw"
        ));
    }

    #[test]
    fn image_and_video_anchors_can_be_expanded_by_the_worker() {
        let spec = MiniMaxM3VisionSpec;
        assert!(spec.worker_expandable(Modality::Image));
        assert!(spec.worker_expandable(Modality::Video));
        assert!(!spec.worker_expandable(Modality::Audio));
    }

    #[test]
    fn declares_image_and_video_limits() {
        let spec = MiniMaxM3VisionSpec;
        let limits = spec.modality_limits(&metadata()).unwrap();

        assert_eq!(limits.get(&Modality::Image), Some(&MAX_IMAGES_PER_REQUEST));
        assert_eq!(MAX_IMAGES_PER_REQUEST, 200);
        assert_eq!(limits.get(&Modality::Video), Some(&MAX_VIDEOS_PER_REQUEST));
        assert_eq!(MAX_VIDEOS_PER_REQUEST, 20);
        assert!(!limits.contains_key(&Modality::Audio));
    }

    #[test]
    fn samples_video_at_the_reference_rate_by_default() {
        assert_eq!(MiniMaxM3VisionSpec.default_video_sample_fps(), Some(1.0));
        assert_eq!(
            MiniMaxM3VisionSpec.video_frame_sampling(),
            FrameSampling::Interval
        );
    }

    #[test]
    fn each_modality_declares_only_its_own_layouts() {
        let spec = MiniMaxM3VisionSpec;

        let image = spec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(image.encoder_input, FieldLayout::flat("patches_per_image"));
        assert!(image.model_specific.contains_key("image_grid_thw"));
        assert!(!image.model_specific.contains_key("video_grid_thw"));

        let video = spec.encoder_field_layouts_for(Modality::Video);
        assert_eq!(video.encoder_input, FieldLayout::flat("patches_per_video"));
        assert!(video.model_specific.contains_key("video_grid_thw"));
        assert!(!video.model_specific.contains_key("patches_per_image"));
    }

    #[test]
    fn clips_sampled_at_different_rates_join_into_one_batch() {
        let spec = MiniMaxM3VisionSpec;
        let layouts = spec
            .encoder_field_layouts_for(Modality::Video)
            .model_specific;

        let clip = |patches: usize, grid_t: i64, seconds: f32| {
            PreprocessedEncoderInputs::new(
                ndarray::Array2::<f32>::zeros((patches, 4)),
                vec![patches],
                vec![(224, 224)],
            )
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![grid_t, 2, 2], 1, 3),
            )
            .with_extra(
                "patches_per_video",
                ModelSpecificValue::int_1d(vec![patches as i64]),
            )
            .with_extra(
                "video_second_per_grid",
                ModelSpecificValue::Tensor {
                    data: vec![seconds],
                    shape: vec![1],
                },
            )
        };

        let joined =
            PreprocessedEncoderInputs::concat(vec![clip(8, 2, 1.0), clip(12, 3, 0.5)], &layouts)
                .expect("clips with their own frame spacing belong in the same batch");

        let spacing = joined.model_specific.get("video_second_per_grid");
        assert!(
            matches!(
                spacing,
                Some(ModelSpecificValue::Tensor { data, shape })
                    if data.as_slice() == [1.0, 0.5] && shape.as_slice() == [2]
            ),
            "expected one row per clip, got {spacing:?}"
        );
    }

    #[test]
    fn each_video_is_laid_out_from_its_own_grid_row() {
        let spec = MiniMaxM3VisionSpec;
        // Two clips joined into one batch: 2 frames of 4 tokens, then 3 frames of 4.
        let preprocessed = preprocessed(vec![8, 12]).with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![2, 4, 4, 3, 4, 4], 2, 3),
        );

        let replacements = spec
            .prompt_replacements_for(&metadata(), &preprocessed, Modality::Video)
            .unwrap();

        assert_eq!(replacements.len(), 2);
        // Each frame is [start] + tokens + [end].
        assert_eq!(replacements[0].tokens.len(), 8 + 2 * 2);
        assert_eq!(replacements[1].tokens.len(), 12 + 2 * 3);
        assert_eq!(
            replacements[0].feature_ranges.as_ref().map(Vec::len),
            Some(2)
        );
        assert_eq!(
            replacements[1].feature_ranges.as_ref().map(Vec::len),
            Some(3)
        );

        // A grid with fewer rows than clips is a broken batch, not a guess.
        let short_grid = self::preprocessed(vec![8, 12]).with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![2, 4, 4], 1, 3),
        );
        assert!(matches!(
            spec.prompt_replacements_for(&metadata(), &short_grid, Modality::Video),
            Err(ModelRegistryError::InvalidPreprocessedField { ref field }) if field == "video_grid_thw"
        ));
    }

    #[test]
    fn audio_is_rejected() {
        let spec = MiniMaxM3VisionSpec;
        let err = spec
            .prompt_replacements_for(&metadata(), &preprocessed(vec![1]), Modality::Audio)
            .unwrap_err();

        assert!(matches!(
            err,
            ModelRegistryError::UnsupportedModality { .. }
        ));
    }

    #[test]
    fn grid_tensors_stay_on_cpu() {
        // vLLM marks both grid tensors keep_on_cpu=True.
        let spec = MiniMaxM3VisionSpec;
        let keys = spec.keep_on_cpu_keys();

        assert!(keys.contains(&"image_grid_thw".to_string()));
        assert!(keys.contains(&"video_grid_thw".to_string()));
    }
}
