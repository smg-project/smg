//! Multimodal processing core: fetch media → preprocess pixels → expand
//! placeholder tokens → build the lightweight [`MultimodalIntermediate`].
//!
//! Protocol adapters first normalize media into a [`MediaPlan`], so this module
//! has one request-independent processing entry point.

use std::{collections::HashMap, sync::Arc, time::Instant};

use anyhow::{Context, Result};
use futures::future::try_join_all;
use llm_multimodal::{
    AsyncMultiModalTracker, AudioClip, EncoderFieldLayouts, ImageFrame, Modality, ModelMetadata,
    ModelProcessorSpec, PlaceholderRange, PreProcessorConfig, PreprocessedEncoderInputs,
    PromptReplacement, TrackedMedia, TrackerOutput, VideoClip, VisionProcessorRegistry,
};
use llm_tokenizer::TokenizerTrait;
use tracing::{debug, info, warn};

use super::{
    config::{MultimodalComponents, MultimodalModelConfig},
    log_mm_timing_enabled,
    pixel_cache::{config_fingerprint, CachedPreprocessedItem, PixelCache, PixelCacheKey},
    plan::MediaPlan,
    MediaBatch, MultimodalIntermediate, MultimodalOutput, PrecomputedMultimodalIntermediate,
    PromptBinding, RegistryTokenizer,
};

struct PreparedMultimodalPart {
    preprocessed: PreprocessedEncoderInputs,
    media: MediaBatch,
    prompt_replacements: Vec<PromptReplacement>,
    search_token_id: Option<u32>,
    placeholder_token_id: Option<u32>,
    field_layouts: EncoderFieldLayouts,
    keep_on_cpu_keys: Vec<String>,
}

/// Process a protocol-independent, ordered media plan.
pub(crate) async fn process_multimodal_plan(
    plan: MediaPlan,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    token_ids: Vec<u32>,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
) -> Result<MultimodalOutput> {
    let log_timing = log_mm_timing_enabled();
    let total_started = Instant::now();
    let media_started = Instant::now();
    let mut tracker = AsyncMultiModalTracker::new(components.media_connector.clone());

    for part in plan.into_parts() {
        tracker
            .push_part(part)
            .map_err(|e| anyhow::anyhow!("Failed to push content part: {e}"))?;
    }

    let tracker_output: TrackerOutput = tracker
        .finalize()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to finalize multimodal tracker: {e}"))?;

    let images: Vec<Arc<ImageFrame>> = tracker_output
        .data
        .get(&Modality::Image)
        .map(|media_vec| {
            media_vec
                .iter()
                .filter_map(|m| match m {
                    TrackedMedia::Image(frame) => Some(frame.clone()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    let videos: Vec<Arc<VideoClip>> = tracker_output
        .data
        .get(&Modality::Video)
        .map(|media_vec| {
            media_vec
                .iter()
                .filter_map(|m| match m {
                    TrackedMedia::Video(clip) => Some(clip.clone()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    let audios: Vec<Arc<AudioClip>> = tracker_output
        .data
        .get(&Modality::Audio)
        .map(|media_vec| {
            media_vec
                .iter()
                .filter_map(|m| match m {
                    TrackedMedia::Audio(clip) => Some(clip.clone()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    let media_elapsed_ms = media_started.elapsed().as_secs_f64() * 1000.0;
    let image_count = images.len();
    let audio_count = audios.len();
    let video_count = videos.len();
    let video_frame_count = videos.first().map_or(0, |video| {
        if video.frames().is_empty() {
            video
                .rgb_video()
                .map_or(0, |rgb_video| rgb_video.frames.len())
        } else {
            video.frames().len()
        }
    });
    let mut media_batches = Vec::with_capacity(3);
    if !images.is_empty() {
        media_batches.push(MediaBatch::Images(images));
    }
    if !videos.is_empty() {
        media_batches.push(MediaBatch::Videos(videos));
    }
    if !audios.is_empty() {
        media_batches.push(MediaBatch::Audios(audios));
    }
    if media_batches.is_empty() {
        return Err(anyhow::anyhow!(
            "No media was successfully fetched for multimodal request"
        ));
    }
    let present_modalities = media_batches
        .iter()
        .map(MediaBatch::modality)
        .collect::<Vec<_>>();

    for batch in &media_batches {
        match batch {
            MediaBatch::Images(images) => {
                debug!(
                    image_count = images.len(),
                    item_sizes = ?images.iter().map(|f| (f.image.width(), f.image.height())).collect::<Vec<_>>(),
                    "Fetched images for multimodal processing"
                );
            }
            MediaBatch::Videos(videos) => {
                debug!(
                    video_count = videos.len(),
                    frame_count = video_frame_count,
                    "Fetched video for multimodal processing"
                );
            }
            MediaBatch::Audios(audios) => {
                debug!(
                    audio_count = audios.len(),
                    "Fetched audios for multimodal processing"
                );
            }
        }
    }

    // Step 2: Resolve model spec and preprocess media.
    let config_started = Instant::now();
    let model_config = components
        .config_registry
        .get_or_load(tokenizer_id, tokenizer_source)
        .await?;
    let model_type = model_config
        .config
        .get("model_type")
        .and_then(|v| v.as_str());
    let registry_tokenizer = RegistryTokenizer(tokenizer);
    let metadata = ModelMetadata {
        model_id,
        tokenizer: &registry_tokenizer,
        config: &model_config.config,
    };
    let spec = components
        .model_registry
        .lookup(&metadata)
        .ok_or_else(|| anyhow::anyhow!("Multimodal not supported for model: {model_id}"))?;
    let config_elapsed_ms = config_started.elapsed().as_secs_f64() * 1000.0;

    let preprocess_started = Instant::now();
    let mut prepared_parts = Vec::with_capacity(media_batches.len());
    // Every modality batch is independent until prompt expansion. Poll all
    // preprocessors concurrently and preserve the batch order in the returned
    // vector so the media/preprocessed zip below remains exact for any model-
    // validated modality combination.
    let preprocessed_parts = try_join_all(media_batches.iter().map(|media| {
        preprocess_modality(
            media,
            components,
            model_id,
            model_type,
            spec,
            tokenizer_id,
            &model_config,
        )
    }))
    .await?;

    for (media, preprocessed) in media_batches.into_iter().zip(preprocessed_parts) {
        let modality = media.modality();
        debug!(
            ?modality,
            item_count = preprocessed.feature_token_counts.len(),
            total_tokens = preprocessed.feature_token_counts.iter().sum::<usize>(),
            "Multimodal preprocessing complete"
        );

        let prompt_replacements = spec
            .prompt_replacements_for(&metadata, &preprocessed, modality)
            .map_err(|e| anyhow::anyhow!("Failed to compute prompt replacements: {e}"))?;

        let media_count = media.len();
        anyhow::ensure!(
            preprocessed.feature_token_counts.len() == media_count,
            "Preprocessing item count mismatch for {modality}: {} media items, {} feature-token counts",
            media_count,
            preprocessed.feature_token_counts.len()
        );
        anyhow::ensure!(
            prompt_replacements.len() == media_count,
            "Prompt replacement count mismatch for {modality}: {} media items, {} replacements",
            media_count,
            prompt_replacements.len()
        );

        // Two token IDs may differ for the same placeholder:
        // - search_token_id: what the tokenizer actually emits (e.g. 200090 for "<|image|>")
        // - placeholder_token_id: what the model config declares (e.g. image_token_id/video_token_id)
        let placeholder_token = spec
            .placeholder_token_for(&metadata, modality)
            .map_err(|e| anyhow::anyhow!("Failed to get placeholder token: {e}"))?;
        let search_token_id = tokenizer.token_to_id(&placeholder_token);
        let placeholder_token_id: Option<u32> = match spec
            .placeholder_token_id_for(&metadata, modality)
        {
            Ok(id) => Some(u32::try_from(id).map_err(|_| {
                anyhow::anyhow!(
                    "Invalid negative placeholder token ID {id} for modality {modality}"
                )
            })?),
            Err(e) => {
                warn!(
                    error = %e,
                    ?search_token_id,
                    "Failed to resolve placeholder_token_id from config, falling back to tokenizer lookup"
                );
                search_token_id
            }
        };

        prepared_parts.push(PreparedMultimodalPart {
            preprocessed,
            media,
            prompt_replacements,
            search_token_id,
            placeholder_token_id,
            field_layouts: spec.encoder_field_layouts_for(modality),
            keep_on_cpu_keys: spec.keep_on_cpu_keys_for(modality),
        });
    }
    let preprocess_elapsed_ms = preprocess_started.elapsed().as_secs_f64() * 1000.0;

    // Step 3: Compute prompt replacements and expand tokens.
    let expansion_started = Instant::now();
    let expansions = prepared_parts
        .iter()
        .map(|part| ModalityExpansion {
            modality: part.media.modality(),
            search_token_id: part.search_token_id,
            placeholder_token_id: part.placeholder_token_id,
            replacements: &part.prompt_replacements,
        })
        .collect::<Vec<_>>();
    let expanded = expand_tokens_for_modalities(&token_ids, &expansions)?;
    let placeholder_count = expanded.bindings.iter().map(Vec::len).sum::<usize>();

    debug!(
        original_len = token_ids.len(),
        expanded_len = expanded.token_ids.len(),
        placeholder_count,
        modality_count = prepared_parts.len(),
        "Token expansion complete"
    );
    let expansion_elapsed_ms = expansion_started.elapsed().as_secs_f64() * 1000.0;
    let original_tokens = token_ids.len();
    let expanded_tokens = expanded.token_ids.len();

    // Step 4: Build lightweight intermediate (defers tensor serialization to assembly)
    let batches = prepared_parts
        .into_iter()
        .zip(expanded.bindings)
        .map(|(part, bindings)| PrecomputedMultimodalIntermediate {
            preprocessed: part.preprocessed,
            media: part.media,
            bindings,
            placeholder_token_id: part.placeholder_token_id,
            field_layouts: part.field_layouts,
            keep_on_cpu_keys: part.keep_on_cpu_keys,
        })
        .collect::<Vec<_>>();
    let intermediate = MultimodalIntermediate::try_new(batches)?;

    if log_timing {
        info!(
            modalities = ?present_modalities,
            image_count,
            audio_count,
            video_count,
            video_frame_count,
            media_fetch_decode_ms = media_elapsed_ms,
            config_lookup_ms = config_elapsed_ms,
            preprocess_ms = preprocess_elapsed_ms,
            token_expand_ms = expansion_elapsed_ms,
            total_ms = total_started.elapsed().as_secs_f64() * 1000.0,
            original_tokens,
            expanded_tokens,
            "smg_mm_timing process_multimodal_plan"
        );
    }

    Ok(MultimodalOutput {
        expanded_token_ids: expanded.token_ids,
        intermediate,
    })
}

async fn preprocess_modality(
    media: &MediaBatch,
    components: &MultimodalComponents,
    model_id: &str,
    model_type: Option<&str>,
    spec: &dyn ModelProcessorSpec,
    tokenizer_id: &str,
    model_config: &MultimodalModelConfig,
) -> Result<PreprocessedEncoderInputs> {
    // Run CPU-intensive preprocessing on a blocking thread pool so it doesn't
    // block the tokio async runtime under concurrent load.
    // TODO: consider making the thread pool size configurable.
    let modality = media.modality();
    let pp_config = match modality {
        Modality::Video => model_config
            .video_preprocessor_config
            .clone()
            .unwrap_or_else(|| model_config.preprocessor_config.clone()),
        _ => model_config.preprocessor_config.clone(),
    };

    if let MediaBatch::Images(images) = media {
        if let (Some(cache), [image]) = (components.pixel_cache.clone(), images.as_slice()) {
            return preprocess_image_cached(
                cache,
                image,
                components.vision_processor_registry.clone(),
                model_id.to_string(),
                model_type.map(String::from),
                pp_config,
                config_fingerprint(tokenizer_id, &model_config.config),
            )
            .await;
        }
    }

    let registry = components.vision_processor_registry.clone();
    let model_id_owned = model_id.to_string();
    let model_type_owned = model_type.map(String::from);
    let media_for_preprocess = media.clone(); // cheap Arc refcount bumps
    let audio_processor = if modality == Modality::Audio {
        Some(
            spec.audio_processor(&model_config.config, &model_config.preprocessor_config)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No audio processor registered for model spec: {}",
                        spec.name()
                    )
                })?,
        )
    } else {
        None
    };

    tokio::task::spawn_blocking(move || match media_for_preprocess {
        MediaBatch::Images(images) => {
            let processor = registry
                .find(&model_id_owned, model_type_owned.as_deref())
                .ok_or_else(|| {
                    anyhow::anyhow!("No vision processor found for model: {model_id_owned}")
                })?;
            // Extract DynamicImages inside the blocking closure so the expensive
            // clone happens off the tokio async runtime.
            let raw_images: Vec<image::DynamicImage> =
                images.iter().map(|frame| frame.image.clone()).collect();
            processor
                .preprocess(&raw_images, &pp_config)
                .map_err(|e| anyhow::anyhow!("Image preprocessing failed: {e}"))
        }
        MediaBatch::Videos(videos) => {
            // VisionPreProcessor currently models one decoded clip per call.
            // Video-capable model specs therefore declare a per-request limit
            // of one; a future batched-video processor can lift this without
            // adding any modality-combination policy here.
            let [video] = videos.as_slice() else {
                anyhow::bail!(
                    "Video preprocessing currently requires exactly one clip per modality batch; got {}",
                    videos.len()
                );
            };
            let processor = registry
                .find(&model_id_owned, model_type_owned.as_deref())
                .ok_or_else(|| {
                    anyhow::anyhow!("No vision processor found for model: {model_id_owned}")
                })?;
            let video_pp_config = with_video_sample_fps(pp_config.clone(), video);

            if !video.frames().is_empty() {
                return processor
                    .preprocess_video(video.frames(), &video_pp_config)
                    .map_err(|e| anyhow::anyhow!("Video preprocessing failed: {e}"));
            }

            if let Some(rgb_video) = video.rgb_video() {
                match rgb_video.frame_refs() {
                    Ok(frame_refs) => match processor
                        .preprocess_video_rgb(&frame_refs, &video_pp_config)
                    {
                        Ok(preprocessed) => return Ok(preprocessed),
                        Err(error) => {
                            warn!(
                                error = %error,
                                "RGB video preprocessing fast path failed; falling back to materialized frames"
                            );
                        }
                    },
                    Err(error) => {
                        warn!(
                            error = %error,
                            "RGB video frame refs are invalid; falling back to materialized frames"
                        );
                    }
                }
            }

            let frames = video
                .materialized_frames()
                .map_err(|e| anyhow::anyhow!("Video frame materialization failed: {e}"))?;
            processor
                .preprocess_video(&frames, &video_pp_config)
                .map_err(|e| anyhow::anyhow!("Video preprocessing failed: {e}"))
        }
        MediaBatch::Audios(audios) => {
            let processor = audio_processor.ok_or_else(|| {
                anyhow::anyhow!("Model did not provide an audio processor")
            })?;
            processor
                .preprocess(&audios)
                .map_err(|e| anyhow::anyhow!("Audio preprocessing failed: {e}"))
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("Preprocessing task panicked: {e}"))?
}

fn with_video_sample_fps(mut config: PreProcessorConfig, video: &VideoClip) -> PreProcessorConfig {
    config
        .extra
        .insert("fps".to_string(), serde_json::json!(video.sample_fps()));
    config
}

/// Pixel-cache image preprocessing for single-image requests.
async fn preprocess_image_cached(
    cache: Arc<PixelCache>,
    image: &Arc<ImageFrame>,
    registry: Arc<VisionProcessorRegistry>,
    model_id: String,
    model_type: Option<String>,
    pp_config: PreProcessorConfig,
    fingerprint: u64,
) -> Result<PreprocessedEncoderInputs> {
    let key = PixelCacheKey {
        image_hash: image.hash.clone(),
        config_fingerprint: fingerprint,
    };
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.preprocessed.clone());
    }

    let preprocessed = preprocess_image_batch(
        registry,
        model_id,
        model_type,
        pp_config,
        std::slice::from_ref(image),
    )
    .await?;
    cache.insert(
        key,
        Arc::new(CachedPreprocessedItem {
            preprocessed: preprocessed.clone(),
        }),
    );
    Ok(preprocessed)
}

async fn preprocess_image_batch(
    registry: Arc<VisionProcessorRegistry>,
    model_id: String,
    model_type: Option<String>,
    pp_config: PreProcessorConfig,
    images: &[Arc<ImageFrame>],
) -> Result<PreprocessedEncoderInputs> {
    let raw_images: Vec<image::DynamicImage> = images.iter().map(|f| f.image.clone()).collect();
    tokio::task::spawn_blocking(move || {
        let processor = registry
            .find(&model_id, model_type.as_deref())
            .ok_or_else(|| anyhow::anyhow!("No vision processor found for model: {model_id}"))?;
        processor
            .preprocess(&raw_images, &pp_config)
            .map_err(|e| anyhow::anyhow!("Image preprocessing failed: {e}"))
    })
    .await
    .map_err(|e| anyhow::anyhow!("Preprocessing task panicked: {e}"))?
}

struct ModalityExpansion<'a> {
    modality: Modality,
    search_token_id: Option<u32>,
    placeholder_token_id: Option<u32>,
    replacements: &'a [PromptReplacement],
}

#[derive(Debug)]
struct ExpandedMultimodalTokens {
    token_ids: Vec<u32>,
    bindings: Vec<Vec<PromptBinding>>,
}

fn expand_tokens_for_modalities(
    token_ids: &[u32],
    expansions: &[ModalityExpansion<'_>],
) -> Result<ExpandedMultimodalTokens> {
    let mut anchor_to_expansion = HashMap::with_capacity(expansions.len());
    for (idx, expansion) in expansions.iter().enumerate() {
        let Some(anchor_id) = expansion.search_token_id else {
            anyhow::ensure!(
                expansion.replacements.is_empty(),
                "Could not resolve prompt anchor token ID for {} ({} replacements)",
                expansion.modality,
                expansion.replacements.len()
            );
            continue;
        };
        if let Some(previous_idx) = anchor_to_expansion.insert(anchor_id, idx) {
            return Err(anyhow::anyhow!(
                "Prompt anchor token ID {anchor_id} is shared by {} and {}; anchors must be unique",
                expansions[previous_idx].modality,
                expansion.modality
            ));
        }
        for (item_index, replacement) in expansion.replacements.iter().enumerate() {
            anyhow::ensure!(
                replacement.modality == expansion.modality,
                "Prompt replacement {item_index} has modality {}, expected {}",
                replacement.modality,
                expansion.modality
            );
            anyhow::ensure!(
                !replacement.tokens.is_empty(),
                "Prompt replacement {item_index} for {} is empty",
                expansion.modality
            );
        }
    }

    let mut expanded = Vec::with_capacity(token_ids.len());
    let mut bindings = vec![Vec::new(); expansions.len()];
    let mut replacement_indices = vec![0usize; expansions.len()];
    let mut prompt_ordinal = 0usize;

    for (prompt_offset, &token) in token_ids.iter().enumerate() {
        if let Some(&idx) = anchor_to_expansion.get(&token) {
            let expansion = &expansions[idx];
            let item_index = replacement_indices[idx];
            let replacement = expansion.replacements.get(item_index).ok_or_else(|| {
                anyhow::anyhow!(
                    "Extra prompt anchor for {} at input token offset {prompt_offset}: expected {} anchors",
                    expansion.modality,
                    expansion.replacements.len()
                )
            })?;
            let replacement_tokens = replacement
                .tokens
                .iter()
                .enumerate()
                .map(|(replacement_offset, &token)| {
                    u32::try_from(token).map_err(|_| {
                        anyhow::anyhow!(
                            "Invalid negative token ID {token} in {} replacement {item_index} at offset {replacement_offset}",
                            expansion.modality
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let offset = expanded.len();
            let length = replacement_tokens.len();
            let patches = if let Some(feature_ranges) = &replacement.feature_ranges {
                explicit_feature_ranges(offset, length, feature_ranges).with_context(|| {
                    format!(
                        "Invalid explicit feature ranges in {} replacement {item_index}",
                        expansion.modality
                    )
                })?
            } else {
                patch_ranges(offset, &replacement_tokens, expansion.placeholder_token_id)
            };
            expanded.extend(replacement_tokens);
            let prefix = replacement.structural_prefix.min(offset);
            bindings[idx].push(PromptBinding {
                item_index,
                prompt_ordinal,
                structural: PlaceholderRange {
                    offset: offset - prefix,
                    length: length + prefix,
                },
                patches,
            });
            prompt_ordinal = prompt_ordinal
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("Prompt binding ordinal overflow"))?;
            replacement_indices[idx] += 1;
        } else {
            expanded.push(token);
        }
    }

    for (idx, expansion) in expansions.iter().enumerate() {
        anyhow::ensure!(
            replacement_indices[idx] == expansion.replacements.len(),
            "Missing prompt anchors for {}: expected {}, found {}",
            expansion.modality,
            expansion.replacements.len(),
            replacement_indices[idx]
        );
    }

    Ok(ExpandedMultimodalTokens {
        token_ids: expanded,
        bindings,
    })
}

fn patch_ranges(
    offset: usize,
    replacement_tokens: &[u32],
    placeholder_token_id: Option<u32>,
) -> Vec<PlaceholderRange> {
    let Some(placeholder_id) = placeholder_token_id else {
        return Vec::new();
    };

    let mut ranges = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, &token) in replacement_tokens.iter().enumerate() {
        let pos = offset + i;
        if token == placeholder_id {
            if run_start.is_none() {
                run_start = Some(pos);
            }
        } else if let Some(start) = run_start.take() {
            ranges.push(PlaceholderRange {
                offset: start,
                length: pos - start,
            });
        }
    }
    if let Some(start) = run_start {
        let end = offset + replacement_tokens.len();
        ranges.push(PlaceholderRange {
            offset: start,
            length: end - start,
        });
    }
    ranges
}

fn explicit_feature_ranges(
    replacement_offset: usize,
    replacement_length: usize,
    relative_ranges: &[PlaceholderRange],
) -> Result<Vec<PlaceholderRange>> {
    anyhow::ensure!(
        !relative_ranges.is_empty(),
        "explicit feature ranges must not be empty"
    );
    let mut absolute_ranges = Vec::with_capacity(relative_ranges.len());
    let mut previous_end = 0usize;
    for (index, range) in relative_ranges.iter().enumerate() {
        anyhow::ensure!(range.length > 0, "feature range {index} has zero length");
        let relative_end = range
            .offset
            .checked_add(range.length)
            .ok_or_else(|| anyhow::anyhow!("feature range {index} overflows usize"))?;
        anyhow::ensure!(
            relative_end <= replacement_length,
            "feature range {index} ({}, {}) exceeds replacement length {replacement_length}",
            range.offset,
            range.length
        );
        anyhow::ensure!(
            index == 0 || range.offset >= previous_end,
            "feature range {index} overlaps or is out of order"
        );
        absolute_ranges.push(PlaceholderRange {
            offset: replacement_offset
                .checked_add(range.offset)
                .ok_or_else(|| {
                    anyhow::anyhow!("feature range {index} absolute offset overflows")
                })?,
            length: range.length,
        });
        previous_end = relative_end;
    }
    Ok(absolute_ranges)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use llm_multimodal::VideoSource;

    use super::*;

    #[test]
    fn decoded_video_sample_fps_overrides_processor_default() {
        let video = VideoClip::new_with_sample_fps(
            Vec::new(),
            Bytes::new(),
            VideoSource::InlineBytes,
            "video-hash".to_string(),
            0.8,
        );

        let config = with_video_sample_fps(PreProcessorConfig::default(), &video);

        assert!((config.get_extra::<f32>("fps").unwrap() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_expand_tokens_basic() {
        let token_ids = vec![1, 2, 100, 3, 4]; // 100 is the placeholder
        let replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![50, 50, 50, 50], // Expand to 4 tokens
            feature_ranges: None,
            structural_prefix: 0,
        }];

        let expansion = ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(100),
            placeholder_token_id: None,
            replacements: &replacements,
        };
        let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

        assert_eq!(result.token_ids, vec![1, 2, 50, 50, 50, 50, 3, 4]);
        assert_eq!(result.bindings[0].len(), 1);
        assert_eq!(result.bindings[0][0].structural.offset, 2);
        assert_eq!(result.bindings[0][0].structural.length, 4);
        assert!(result.bindings[0][0].patches.is_empty());
    }

    #[test]
    fn test_expand_tokens_structural_prefix_folds_leading_marker() {
        // Qwen video: the chat template emits <VS><video_pad><VE> (777, 100, 778).
        // The replacement declares structural_prefix=1, so the reported range must
        // start on the leading <VS> (offset 2, not 3) and grow by one, letting a
        // backend that scans the range for <VS> (vLLM's per-frame video mrope) find
        // it. The token stream is unchanged — the marker is not re-emitted.
        let token_ids = vec![1, 2, 777, 100, 778, 3]; // 100 is the placeholder
        let replacements = vec![PromptReplacement {
            modality: Modality::Video,
            placeholder_token: "<|video_pad|>".to_string(),
            tokens: vec![50, 50, 50], // expands to 3 video tokens
            feature_ranges: None,
            structural_prefix: 1,
        }];

        let expansion = ModalityExpansion {
            modality: Modality::Video,
            search_token_id: Some(100),
            placeholder_token_id: None,
            replacements: &replacements,
        };
        let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

        assert_eq!(result.token_ids, vec![1, 2, 777, 50, 50, 50, 778, 3]);
        assert_eq!(result.bindings[0].len(), 1);
        // Range starts on <VS> (index 2) and covers it + the 3 video tokens.
        assert_eq!(result.bindings[0][0].structural.offset, 2);
        assert_eq!(result.bindings[0][0].structural.length, 4);
    }

    #[test]
    fn test_expand_tokens_no_placeholder() {
        let token_ids = vec![1, 2, 3];
        let replacements = Vec::new();
        let expansion = ModalityExpansion {
            modality: Modality::Image,
            search_token_id: None,
            placeholder_token_id: None,
            replacements: &replacements,
        };
        let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

        assert_eq!(result.token_ids, vec![1, 2, 3]);
        assert!(result.bindings[0].is_empty());
    }

    #[test]
    fn test_expand_tokens_multiple_images() {
        let token_ids = vec![1, 100, 2, 100, 3]; // Two placeholder tokens
        let replacements = vec![
            PromptReplacement {
                modality: Modality::Image,
                placeholder_token: "<image>".to_string(),
                tokens: vec![50, 50], // 2 tokens for first image
                feature_ranges: None,
                structural_prefix: 0,
            },
            PromptReplacement {
                modality: Modality::Image,
                placeholder_token: "<image>".to_string(),
                tokens: vec![60, 60, 60], // 3 tokens for second image
                feature_ranges: None,
                structural_prefix: 0,
            },
        ];

        let expansion = ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(100),
            placeholder_token_id: None,
            replacements: &replacements,
        };
        let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

        assert_eq!(result.token_ids, vec![1, 50, 50, 2, 60, 60, 60, 3]);
        assert_eq!(result.bindings[0].len(), 2);
        assert_eq!(result.bindings[0][0].structural.offset, 1);
        assert_eq!(result.bindings[0][0].structural.length, 2);
        assert_eq!(result.bindings[0][1].structural.offset, 4);
        assert_eq!(result.bindings[0][1].structural.length, 3);
    }

    #[test]
    fn test_expand_tokens_patch_offsets_with_structural() {
        // Simulates Llama-4: placeholder expands to structural + patch tokens
        // 88=image_start, 92=patch(im_token_id), 93=separator, 89=image_end
        let token_ids = vec![1, 100, 2]; // 100 is the placeholder
        let replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![88, 92, 92, 92, 93, 92, 92, 92, 89], // start + patches + sep + patches + end
            feature_ranges: None,
            structural_prefix: 0,
        }];

        let expansion = ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(100),
            placeholder_token_id: Some(92),
            replacements: &replacements,
        };
        let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

        // Full structural range
        assert_eq!(result.bindings[0].len(), 1);
        assert_eq!(result.bindings[0][0].structural.offset, 1);
        assert_eq!(result.bindings[0][0].structural.length, 9);

        // Patch-only offsets: two runs of token 92
        let patch = &result.bindings[0][0].patches;
        assert_eq!(patch.len(), 2);
        assert_eq!((patch[0].offset, patch[0].length), (2, 3));
        assert_eq!((patch[1].offset, patch[1].length), (6, 3));
    }

    #[test]
    fn test_explicit_feature_range_excludes_same_id_structural_header() {
        let replacements = vec![PromptReplacement::sequence(
            Modality::Image,
            "<|content_image|>",
            vec![200005, 200005, 200005, 200005],
        )
        .with_feature_span(1, 3)];
        let expansion = ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(200005),
            placeholder_token_id: Some(200005),
            replacements: &replacements,
        };

        let result = expand_tokens_for_modalities(&[1, 200005, 2], &[expansion]).unwrap();

        assert_eq!(result.token_ids, vec![1, 200005, 200005, 200005, 200005, 2]);
        assert_eq!(result.bindings[0][0].structural.offset, 1);
        assert_eq!(result.bindings[0][0].structural.length, 4);
        assert_eq!(result.bindings[0][0].patches.len(), 1);
        assert_eq!(result.bindings[0][0].patches[0].offset, 2);
        assert_eq!(result.bindings[0][0].patches[0].length, 3);
    }

    #[test]
    fn test_explicit_feature_range_must_fit_replacement() {
        let replacements =
            vec![
                PromptReplacement::sequence(Modality::Image, "<image>", vec![50, 50])
                    .with_feature_span(1, 2),
            ];
        let expansion = ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(100),
            placeholder_token_id: Some(50),
            replacements: &replacements,
        };

        let error = expand_tokens_for_modalities(&[100], &[expansion]).unwrap_err();
        assert!(format!("{error:#}").contains("exceeds replacement length"));
    }

    #[test]
    fn test_expand_tokens_preserves_template_owned_audio_end() {
        // The template owns the audio end marker. Expansion preserves the
        // anchor, adds feature tokens, and must not inject another end marker.
        let audio_anchor: u32 = 100;
        let audio_placeholder: u32 = 101;
        let audio_end: u32 = 102;
        let message_end: u32 = 103;
        let token_ids = vec![1, audio_anchor, audio_end, message_end];
        let replacements = vec![PromptReplacement {
            modality: Modality::Audio,
            placeholder_token: "<audio>".to_string(),
            tokens: vec![
                audio_anchor as i32,
                audio_placeholder as i32,
                audio_placeholder as i32,
            ],
            feature_ranges: None,
            structural_prefix: 0,
        }];

        let expansion = ModalityExpansion {
            modality: Modality::Audio,
            search_token_id: Some(audio_anchor),
            placeholder_token_id: Some(audio_placeholder),
            replacements: &replacements,
        };
        let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

        assert_eq!(
            result.token_ids,
            vec![
                1,
                audio_anchor,
                audio_placeholder,
                audio_placeholder,
                audio_end,
                message_end
            ]
        );
        assert_eq!(
            result
                .token_ids
                .iter()
                .filter(|&&token| token == audio_end)
                .count(),
            1
        );
        assert_eq!(result.bindings[0][0].patches.len(), 1);
        assert_eq!(result.bindings[0][0].patches[0].offset, 2);
        assert_eq!(result.bindings[0][0].patches[0].length, 2);
    }

    #[test]
    fn test_expand_tokens_mixed_image_audio_offsets_follow_final_prompt() {
        let audio_anchor: u32 = 100;
        let audio_placeholder: u32 = 101;
        let audio_end: u32 = 102;
        let image_anchor: u32 = 200;
        let image_placeholder: u32 = 201;
        let message_end: u32 = 103;
        let token_ids = vec![
            1,
            audio_anchor,
            audio_end,
            message_end,
            2,
            image_anchor,
            message_end,
        ];
        let audio_replacements = vec![PromptReplacement {
            modality: Modality::Audio,
            placeholder_token: "<audio>".to_string(),
            tokens: vec![
                audio_anchor as i32,
                audio_placeholder as i32,
                audio_placeholder as i32,
            ],
            feature_ranges: None,
            structural_prefix: 0,
        }];
        let image_replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![
                image_anchor as i32,
                image_placeholder as i32,
                image_placeholder as i32,
                image_placeholder as i32,
            ],
            feature_ranges: None,
            structural_prefix: 0,
        }];
        let expansions = vec![
            ModalityExpansion {
                modality: Modality::Image,
                search_token_id: Some(image_anchor),
                placeholder_token_id: Some(image_placeholder),
                replacements: &image_replacements,
            },
            ModalityExpansion {
                modality: Modality::Audio,
                search_token_id: Some(audio_anchor),
                placeholder_token_id: Some(audio_placeholder),
                replacements: &audio_replacements,
            },
        ];

        let result = expand_tokens_for_modalities(&token_ids, &expansions).unwrap();

        assert_eq!(
            result.token_ids,
            vec![
                1,
                audio_anchor,
                audio_placeholder,
                audio_placeholder,
                audio_end,
                message_end,
                2,
                image_anchor,
                image_placeholder,
                image_placeholder,
                image_placeholder,
                message_end,
            ]
        );
        assert_eq!(result.bindings[0][0].structural.offset, 7);
        assert_eq!(result.bindings[0][0].structural.length, 4);
        assert_eq!(result.bindings[0][0].patches[0].offset, 8);
        assert_eq!(result.bindings[0][0].patches[0].length, 3);
        assert_eq!(result.bindings[0][0].prompt_ordinal, 1);
        assert_eq!(result.bindings[1][0].structural.offset, 1);
        assert_eq!(result.bindings[1][0].structural.length, 3);
        assert_eq!(result.bindings[1][0].patches[0].offset, 2);
        assert_eq!(result.bindings[1][0].patches[0].length, 2);
        assert_eq!(result.bindings[1][0].prompt_ordinal, 0);
    }

    #[test]
    fn test_expand_tokens_three_modalities_follow_final_prompt() {
        let image_replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![110, 101, 111],
            feature_ranges: None,
            structural_prefix: 0,
        }];
        let video_replacements = vec![PromptReplacement {
            modality: Modality::Video,
            placeholder_token: "<video>".to_string(),
            tokens: vec![210, 201, 201, 211],
            feature_ranges: None,
            structural_prefix: 0,
        }];
        let audio_replacements = vec![PromptReplacement {
            modality: Modality::Audio,
            placeholder_token: "<audio>".to_string(),
            tokens: vec![310, 301, 301],
            feature_ranges: None,
            structural_prefix: 0,
        }];
        let expansions = vec![
            ModalityExpansion {
                modality: Modality::Image,
                search_token_id: Some(100),
                placeholder_token_id: Some(101),
                replacements: &image_replacements,
            },
            ModalityExpansion {
                modality: Modality::Video,
                search_token_id: Some(200),
                placeholder_token_id: Some(201),
                replacements: &video_replacements,
            },
            ModalityExpansion {
                modality: Modality::Audio,
                search_token_id: Some(300),
                placeholder_token_id: Some(301),
                replacements: &audio_replacements,
            },
        ];

        let result = expand_tokens_for_modalities(&[300, 9, 100, 8, 200], &expansions).unwrap();

        assert_eq!(
            result.token_ids,
            vec![310, 301, 301, 9, 110, 101, 111, 8, 210, 201, 201, 211]
        );
        let image = &result.bindings[0][0];
        let video = &result.bindings[1][0];
        let audio = &result.bindings[2][0];
        assert_eq!(image.prompt_ordinal, 1);
        assert_eq!((image.structural.offset, image.patches[0].offset), (4, 5));
        assert_eq!(video.prompt_ordinal, 2);
        assert_eq!((video.structural.offset, video.patches[0].offset), (8, 9));
        assert_eq!(audio.prompt_ordinal, 0);
        assert_eq!((audio.structural.offset, audio.patches[0].offset), (0, 1));
    }

    #[test]
    fn test_expand_tokens_more_placeholders_than_replacements() {
        // Two prompt anchors for one replacement must fail instead of silently
        // binding user/template text to the wrong media item.
        let token_ids = vec![1, 100, 2, 100, 3];
        let replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![50, 50],
            feature_ranges: None,
            structural_prefix: 0,
        }];

        let expansion = ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(100),
            placeholder_token_id: None,
            replacements: &replacements,
        };
        let error = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap_err();
        assert!(error.to_string().contains("Extra prompt anchor"));
    }

    #[test]
    fn test_expand_tokens_rejects_missing_anchor() {
        let replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![50, 50],
            feature_ranges: None,
            structural_prefix: 0,
        }];

        let expansion = ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(100),
            placeholder_token_id: None,
            replacements: &replacements,
        };
        let error = expand_tokens_for_modalities(&[1, 2, 3], &[expansion]).unwrap_err();
        assert!(error.to_string().contains("Missing prompt anchors"));
    }

    #[test]
    fn test_expand_tokens_rejects_shared_anchor_id() {
        let image_replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![50],
            feature_ranges: None,
            structural_prefix: 0,
        }];
        let audio_replacements = vec![PromptReplacement {
            modality: Modality::Audio,
            placeholder_token: "<audio>".to_string(),
            tokens: vec![60],
            feature_ranges: None,
            structural_prefix: 0,
        }];
        let expansions = [
            ModalityExpansion {
                modality: Modality::Image,
                search_token_id: Some(100),
                placeholder_token_id: Some(50),
                replacements: &image_replacements,
            },
            ModalityExpansion {
                modality: Modality::Audio,
                search_token_id: Some(100),
                placeholder_token_id: Some(60),
                replacements: &audio_replacements,
            },
        ];

        let error = expand_tokens_for_modalities(&[100, 100], &expansions).unwrap_err();
        assert!(error.to_string().contains("anchors must be unique"));
    }

    #[test]
    fn test_expand_tokens_rejects_negative_replacement_token() {
        let replacements = vec![PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![50, -1],
            feature_ranges: None,
            structural_prefix: 0,
        }];

        let expansion = ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(100),
            placeholder_token_id: Some(50),
            replacements: &replacements,
        };
        let error = expand_tokens_for_modalities(&[100], &[expansion]).unwrap_err();
        assert!(error.to_string().contains("Invalid negative token ID"));
    }
}
