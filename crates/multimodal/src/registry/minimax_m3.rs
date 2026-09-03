use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId},
};

/// Maximum images accepted in one request (MiniMax-M3 spec 1.3.6).
const MAX_IMAGES_PER_REQUEST: usize = 200;

/// Maximum videos accepted in one request (MiniMax-M3 spec 1.3.6).
const MAX_VIDEOS_PER_REQUEST: usize = 20;

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
/// The start/end markers are modality-specific: M3's vocabulary carries a
/// separate `]<]start of video[>[` / `]<]end of video[>[` pair alongside the
/// image one, unlike Qwen's modality-neutral `<|vision_start|>`.
pub(super) struct MiniMaxM3VisionSpec;

impl MiniMaxM3VisionSpec {
    const IMAGE_TOKEN: &'static str = "]<]image[>[";
    const VIDEO_TOKEN: &'static str = "]<]video[>[";
    const IMAGE_START_TOKEN: &'static str = "]<]start of image[>[";
    const IMAGE_END_TOKEN: &'static str = "]<]end of image[>[";
    const VIDEO_START_TOKEN: &'static str = "]<]start of video[>[";
    const VIDEO_END_TOKEN: &'static str = "]<]end of video[>[";

    /// The structural markers wrapping one modality's feature run.
    fn wrapper_tokens(modality: Modality) -> RegistryResult<(&'static str, &'static str)> {
        match modality {
            Modality::Image => Ok((Self::IMAGE_START_TOKEN, Self::IMAGE_END_TOKEN)),
            Modality::Video => Ok((Self::VIDEO_START_TOKEN, Self::VIDEO_END_TOKEN)),
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

    /// Temporal grid depth for one video, from `video_grid_thw[0]`.
    fn video_grid_t(preprocessed: &PreprocessedEncoderInputs) -> Option<usize> {
        match preprocessed.model_specific.get("video_grid_thw") {
            Some(ModelSpecificValue::IntTensor { data, shape })
                if shape == &[1, 3] && !data.is_empty() =>
            {
                usize::try_from(data[0]).ok()
            }
            _ => None,
        }
    }

    /// Build the per-frame video body.
    ///
    /// M3 lays video out as one `]<]start of video[>[` .. `]<]end of video[>[`
    /// block **per temporal frame**, each holding `grid_h * grid_w / merge^2`
    /// pad tokens — not one flat block over the whole clip. vLLM's
    /// `_get_prompt_updates` builds the same shape:
    ///
    /// ```text
    /// for frame in 0..grid_t:
    ///     [start] + [video_token] * M + [end]
    /// ```
    ///
    /// vLLM additionally prefixes each frame with a `]<]X.X seconds[>[` marker
    /// when the sampled frame indices and fps are known, and documents the
    /// no-metadata path as an aligned fallback. SMG does not carry that
    /// per-frame metadata, so the timestamp markers are omitted and the frame
    /// blocks alone are emitted.
    ///
    /// Returns `None` when the layout cannot apply (unknown or single frame, or
    /// a token count that does not divide evenly), leaving the caller on the
    /// single-block path.
    fn per_frame_video_tokens(
        metadata: &ModelMetadata,
        pad_token_id: TokenId,
        num_tokens: usize,
        grid_t: usize,
    ) -> RegistryResult<Option<Vec<TokenId>>> {
        if grid_t <= 1 || num_tokens == 0 || !num_tokens.is_multiple_of(grid_t) {
            return Ok(None);
        }
        let start_id = metadata.token_id(Self::VIDEO_START_TOKEN)?;
        let end_id = metadata.token_id(Self::VIDEO_END_TOKEN)?;

        let per_frame = num_tokens / grid_t;
        let mut tokens = Vec::with_capacity(num_tokens + 2 * grid_t);
        for _ in 0..grid_t {
            tokens.push(start_id);
            tokens.extend(std::iter::repeat_n(pad_token_id, per_frame));
            tokens.push(end_id);
        }
        Ok(Some(tokens))
    }

    /// Feature ranges for a per-frame video body: the pad run inside each
    /// frame's markers.
    fn per_frame_feature_ranges(grid_t: usize, per_frame: usize) -> Vec<PlaceholderRange> {
        (0..grid_t)
            .map(|frame| PlaceholderRange {
                // Each frame contributes [start] + per_frame pads + [end].
                offset: frame * (per_frame + 2) + 1,
                length: per_frame,
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
                let pad_token_id = Self::video_token_id(metadata)?;
                let placeholder_token = self.placeholder_token_for(metadata, Modality::Video)?;
                let grid_t = Self::video_grid_t(preprocessed);

                preprocessed
                    .feature_token_counts
                    .iter()
                    .map(|&num_tokens| {
                        let per_frame = grid_t.and_then(|grid_t| {
                            Self::per_frame_video_tokens(metadata, pad_token_id, num_tokens, grid_t)
                                .transpose()
                                .map(|tokens| tokens.map(|tokens| (tokens, grid_t)))
                        });

                        match per_frame {
                            Some(Ok((tokens, grid_t))) => Ok(PromptReplacement::sequence(
                                Modality::Video,
                                &placeholder_token,
                                tokens,
                            )
                            .with_feature_ranges(Self::per_frame_feature_ranges(
                                grid_t,
                                num_tokens / grid_t,
                            ))),
                            Some(Err(err)) => Err(err),
                            None => Self::wrapped_replacement(
                                metadata,
                                Modality::Video,
                                &placeholder_token,
                                pad_token_id,
                                num_tokens,
                            ),
                        }
                    })
                    .collect()
            }
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Mirrors vLLM's `_get_mm_fields_config` for M3: the pixel tensors are
        // flat over patches and sliced per item by the grid product, while the
        // grid triples are batched one row per item.
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
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        // vLLM marks both grid tensors keep_on_cpu=True.
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::registry::{ModelMetadata, Tokenizer};

    /// Vocabulary ids for M3's media markers, as the checkpoint declares them.
    const IMAGE_ID: TokenId = 200_025;
    const VIDEO_ID: TokenId = 200_026;
    const IMAGE_START_ID: TokenId = 200_029;
    const IMAGE_END_ID: TokenId = 200_030;
    const VIDEO_START_ID: TokenId = 200_031;
    const VIDEO_END_ID: TokenId = 200_032;

    struct M3Tokenizer;

    impl Tokenizer for M3Tokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                MiniMaxM3VisionSpec::IMAGE_TOKEN => Some(IMAGE_ID as u32),
                MiniMaxM3VisionSpec::VIDEO_TOKEN => Some(VIDEO_ID as u32),
                MiniMaxM3VisionSpec::IMAGE_START_TOKEN => Some(IMAGE_START_ID as u32),
                MiniMaxM3VisionSpec::IMAGE_END_TOKEN => Some(IMAGE_END_ID as u32),
                MiniMaxM3VisionSpec::VIDEO_START_TOKEN => Some(VIDEO_START_ID as u32),
                MiniMaxM3VisionSpec::VIDEO_END_TOKEN => Some(VIDEO_END_ID as u32),
                _ => None,
            }
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            match id {
                id if id == IMAGE_ID as u32 => Some(MiniMaxM3VisionSpec::IMAGE_TOKEN.to_string()),
                id if id == VIDEO_ID as u32 => Some(MiniMaxM3VisionSpec::VIDEO_TOKEN.to_string()),
                _ => None,
            }
        }

        fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
            self.token_to_id(text).map(|id| vec![id])
        }
    }

    fn metadata() -> ModelMetadata<'static> {
        static CONFIG: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
        static TOKENIZER: M3Tokenizer = M3Tokenizer;
        let config = CONFIG.get_or_init(|| {
            json!({
                "model_type": "minimax_m3_vl",
                "image_token_index": IMAGE_ID,
                "video_token_index": VIDEO_ID,
            })
        });
        ModelMetadata {
            model_id: "MiniMaxAI/MiniMax-M3",
            config,
            tokenizer: &TOKENIZER,
        }
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
            .prompt_replacements_for(&metadata(), &preprocessed(vec![3]), Modality::Video)
            .unwrap();

        // Video uses M3's own video markers, not the image pair.
        assert_eq!(
            replacements[0].tokens,
            vec![VIDEO_START_ID, VIDEO_ID, VIDEO_ID, VIDEO_ID, VIDEO_END_ID]
        );
        assert_eq!(replacements[0].modality, Modality::Video);
        assert_eq!(replacements[0].placeholder_token, "]<]video[>[");
    }

    /// Preprocessed video carrying a `video_grid_thw` of `[grid_t, h, w]`.
    fn preprocessed_video(counts: Vec<usize>, grid_t: i64) -> PreprocessedEncoderInputs {
        preprocessed(counts).with_extra(
            "video_grid_thw",
            ModelSpecificValue::IntTensor {
                data: vec![grid_t, 4, 4],
                shape: vec![1, 3],
            },
        )
    }

    #[tokio::test]
    async fn multi_frame_video_emits_one_block_per_frame() {
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
        // Each frame is [start] + 4 pads + [end]; vLLM builds the same shape.
        let frame = |_| {
            let mut v = vec![VIDEO_START_ID];
            v.extend(std::iter::repeat_n(VIDEO_ID, 4));
            v.push(VIDEO_END_ID);
            v
        };
        let expected: Vec<TokenId> = (0..3).flat_map(frame).collect();
        assert_eq!(tokens, &expected);
        assert_eq!(tokens.len(), 3 * (4 + 2));
    }

    #[tokio::test]
    async fn multi_frame_feature_ranges_skip_each_frames_markers() {
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

    #[tokio::test]
    async fn single_frame_video_stays_one_block() {
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
                VIDEO_START_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_END_ID
            ]
        );
    }

    #[tokio::test]
    async fn ragged_token_count_falls_back_to_one_block() {
        let spec = MiniMaxM3VisionSpec;
        // 10 tokens over 3 frames does not divide evenly.
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![10], 3),
                Modality::Video,
            )
            .unwrap();

        assert_eq!(replacements[0].tokens.len(), 10 + 2);
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
