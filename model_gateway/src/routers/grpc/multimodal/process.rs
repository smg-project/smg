//! Multimodal processing core: fetch media → preprocess pixels → expand
//! placeholder tokens → build the lightweight [`MultimodalIntermediate`].
//!
//! Protocol adapters first normalize media into a [`MediaPlan`], so this module
//! has one request-independent processing entry point.

use std::{collections::HashMap, sync::Arc, time::Instant};

use anyhow::{Context, Result};
use futures::{future::try_join_all, StreamExt, TryStreamExt};
use llm_multimodal::{
    AsyncMultiModalTracker, AudioClip, EncoderDtype, EncoderFieldLayouts, ImageFrame,
    MediaItemInfo, Modality, ModelMetadata, ModelProcessorSpec, PlaceholderRange,
    PreProcessorConfig, PreprocessedEncoderInputs, PromptReplacement, TrackedMedia, TrackerOutput,
    TransparencyPolicy, VideoClip, VideoClipOptions, VisionPreProcessor,
};
use llm_tokenizer::TokenizerTrait;
use rayon::prelude::*;
use tracing::{debug, info, warn};

use super::{
    config::{MultimodalComponents, MultimodalModelConfig, VisionProcessors},
    encoder_budget::check_item_feature_budget,
    log_mm_timing_enabled,
    pixel_cache::{config_fingerprint, CachedPreprocessedItem, PixelCache, PixelCacheKey},
    plan::MediaPlan,
    transparency::frame_transparency,
    MediaBatch, MultimodalIntermediate, MultimodalOutput, PrecomputedMultimodalIntermediate,
    PromptBinding, RegistryTokenizer,
};
use crate::observability::metrics::Metrics;

struct PreparedMultimodalPart {
    preprocessed: PreprocessedEncoderInputs,
    media: MediaBatch,
    prompt_replacements: Vec<PromptReplacement>,
    search_token_id: Option<u32>,
    placeholder_token_id: Option<u32>,
    field_layouts: EncoderFieldLayouts,
    keep_on_cpu_keys: Vec<String>,
    encoder_input_key: Option<String>,
}

/// What every step of a plan's processing reads: the components, the model
/// and its type, the processor spec and the model config the registry
/// resolved for it with the vision processors the spec built from it, the
/// tokenizer the cache keys by, and the wire's dtype and transparency policy.
struct PlanContext<'a> {
    components: &'a MultimodalComponents,
    model_id: &'a str,
    model_type: Option<&'a str>,
    spec: &'a dyn ModelProcessorSpec,
    tokenizer_id: &'a str,
    model_config: &'a MultimodalModelConfig,
    vision_processors: Arc<VisionProcessors>,
    image_wire_dtype: Option<EncoderDtype>,
    transparency: TransparencyPolicy,
    /// The engine's per-item bound in feature tokens, when known (see
    /// [`super::encoder_budget`]).
    encoder_budget: Option<usize>,
}

/// The milliseconds each step of a plan's processing took, filled by the
/// processor as it goes and read by its timing log.
#[derive(Default)]
struct Timings {
    config_lookup_ms: f64,
    media_fetch_decode_ms: f64,
    preprocess_ms: f64,
    token_expand_ms: f64,
}

fn elapsed_ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// What the tracker fetched and decoded for the plan, per modality, in the
/// plan's order.
struct Fetched {
    images: Vec<Arc<ImageFrame>>,
    videos: Vec<Arc<VideoClip>>,
    audios: Vec<Arc<AudioClip>>,
}

/// The counts the timing log reports.
struct MediaCounts {
    images: usize,
    videos: usize,
    audios: usize,
    video_frames: usize,
}

impl Fetched {
    /// The tracker's media of one modality, in order; `pick` reads the variant.
    fn tracked<T>(
        output: &TrackerOutput,
        modality: Modality,
        pick: fn(&TrackedMedia) -> Option<&Arc<T>>,
    ) -> Vec<Arc<T>> {
        output
            .data
            .get(&modality)
            .map(|media_vec| media_vec.iter().filter_map(pick).cloned().collect())
            .unwrap_or_default()
    }

    fn of(output: &TrackerOutput) -> Self {
        Self {
            images: Self::tracked(output, Modality::Image, |m| match m {
                TrackedMedia::Image(frame) => Some(frame),
                _ => None,
            }),
            videos: Self::tracked(output, Modality::Video, |m| match m {
                TrackedMedia::Video(clip) => Some(clip),
                _ => None,
            }),
            audios: Self::tracked(output, Modality::Audio, |m| match m {
                TrackedMedia::Audio(clip) => Some(clip),
                _ => None,
            }),
        }
    }

    fn counts(&self) -> MediaCounts {
        MediaCounts {
            images: self.images.len(),
            videos: self.videos.len(),
            audios: self.audios.len(),
            video_frames: self.videos.first().map_or(0, |video| {
                if video.frames().is_empty() {
                    video
                        .rgb_video()
                        .map_or(0, |rgb_video| rgb_video.frames.len())
                } else {
                    video.frames().len()
                }
            }),
        }
    }

    /// One batch per modality present, images then videos then audios; a
    /// plan whose fetch yielded nothing is an error.
    fn batches(self) -> Result<Vec<MediaBatch>> {
        let mut media_batches = Vec::with_capacity(3);
        if !self.images.is_empty() {
            media_batches.push(MediaBatch::Images(self.images));
        }
        if !self.videos.is_empty() {
            media_batches.push(MediaBatch::Videos(self.videos));
        }
        if !self.audios.is_empty() {
            media_batches.push(MediaBatch::Audios(self.audios));
        }
        if media_batches.is_empty() {
            return Err(anyhow::anyhow!(
                "No media was successfully fetched for multimodal request"
            ));
        }
        Ok(media_batches)
    }
}

/// Process a protocol-independent, ordered media plan.
///
/// `encoder_budget` is the engine's per-item bound in feature tokens, when
/// known (see [`super::encoder_budget`]): an item above it is refused once
/// its features are counted, before anything reaches the engine.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn process_multimodal_plan(
    plan: MediaPlan,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    token_ids: Vec<u32>,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
    image_wire_dtype: Option<EncoderDtype>,
    transparency: TransparencyPolicy,
    encoder_budget: Option<usize>,
) -> Result<MultimodalOutput> {
    let log_timing = log_mm_timing_enabled();
    let total_started = Instant::now();
    let mut timings = Timings::default();

    // Step 1: Resolve the model spec; it decides how media is fetched.
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
    let vision_processors = model_config.vision_processors(spec);
    // The frames decode under the destination engine's transparency policy,
    // unless the processor takes the image as decoded and handles alpha itself.
    let transparency = frame_transparency(
        vision_processors.image().ok().map(|processor| &**processor),
        transparency,
    );
    timings.config_lookup_ms = elapsed_ms(config_started);
    let ctx = PlanContext {
        components,
        model_id,
        model_type,
        spec,
        tokenizer_id,
        model_config: &model_config,
        vision_processors,
        image_wire_dtype,
        transparency,
        encoder_budget,
    };

    let media_started = Instant::now();
    let fetched = fetch_media(plan, &ctx).await?;
    timings.media_fetch_decode_ms = elapsed_ms(media_started);
    let counts = fetched.counts();
    let media_batches = fetched.batches()?;
    let present_modalities = media_batches
        .iter()
        .map(MediaBatch::modality)
        .collect::<Vec<_>>();
    log_fetched(&media_batches, counts.video_frames);

    // Step 2: Preprocess media.
    let preprocess_started = Instant::now();
    let prepared_parts = prepare_parts(media_batches, &ctx, &metadata, tokenizer).await?;
    timings.preprocess_ms = elapsed_ms(preprocess_started);

    // Step 3: Compute prompt replacements and expand tokens.
    let expansion_started = Instant::now();
    let expanded = expand_prompt(&token_ids, &prepared_parts)?;
    timings.token_expand_ms = elapsed_ms(expansion_started);
    let original_tokens = token_ids.len();
    let expanded_tokens = expanded.token_ids.len();

    // Step 4: Build lightweight intermediate (defers tensor serialization to assembly)
    let intermediate = intermediate_of(prepared_parts, expanded.bindings, spec)?;

    // The stages' time as series, from the clock reads the timing log makes.
    let total_ms = elapsed_ms(total_started);
    for (stage, ms) in [
        ("config_lookup", timings.config_lookup_ms),
        ("media_fetch_decode", timings.media_fetch_decode_ms),
        ("preprocess", timings.preprocess_ms),
        ("token_expand", timings.token_expand_ms),
        ("total", total_ms),
    ] {
        Metrics::record_mm_processing_stage(model_id, stage, ms / 1000.0);
    }
    if log_timing {
        info!(
            modalities = ?present_modalities,
            image_count = counts.images,
            audio_count = counts.audios,
            video_count = counts.videos,
            video_frame_count = counts.video_frames,
            media_fetch_decode_ms = timings.media_fetch_decode_ms,
            config_lookup_ms = timings.config_lookup_ms,
            preprocess_ms = timings.preprocess_ms,
            token_expand_ms = timings.token_expand_ms,
            total_ms,
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

/// Fetch and decode the plan's media through the tracker, under the spec's
/// video sampling and the frames' transparency policy.
async fn fetch_media(plan: MediaPlan, ctx: &PlanContext<'_>) -> Result<Fetched> {
    let video_config = ctx.model_config.video_preprocessor_config.as_ref();
    let mut tracker = AsyncMultiModalTracker::new(ctx.components.media_connector.clone())
        .with_drain_on_error(ctx.components.drain_fetches_on_error)
        .with_fetch_observer(
            Metrics::interned_model_label(ctx.model_id),
            Metrics::record_mm_fetch,
        )
        .with_default_video_sample_fps(ctx.spec.default_video_sample_fps_with(video_config))
        .with_video_frame_sampling(
            ctx.spec
                .video_frame_sampling_with(video_config)
                .with_frame_budget(ctx.components.video_frame_budget),
        )
        .with_image_transparency(ctx.transparency)
        .with_minimax_image_resize(
            ctx.spec.name() == "minimax_m3"
                && !ctx
                    .model_config
                    .preprocessor_config
                    .get_extra::<bool>(llm_multimodal::minimax_resize::SGLANG_HF_RESIZE_KEY)
                    .unwrap_or(false),
        );

    for part in plan.into_parts() {
        tracker
            .push_part(part)
            .context("Failed to push content part")?;
    }

    let tracker_output: TrackerOutput = tracker
        .finalize()
        .await
        .context("Failed to finalize multimodal tracker")?;
    Ok(Fetched::of(&tracker_output))
}

/// What was fetched, per batch, in the debug log.
fn log_fetched(media_batches: &[MediaBatch], video_frame_count: usize) {
    for batch in media_batches {
        match batch {
            MediaBatch::Images(images) => {
                debug!(
                    image_count = images.len(),
                    item_sizes = ?images.iter().map(|f| f.size()).collect::<Vec<_>>(),
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
}

/// Preprocess every batch and pair each with its prompt replacements and
/// placeholder tokens. Every modality batch is independent until prompt
/// expansion: all preprocessors are polled concurrently and the batch order
/// is preserved, so the media/preprocessed zip stays exact for any
/// model-validated modality combination.
async fn prepare_parts(
    media_batches: Vec<MediaBatch>,
    ctx: &PlanContext<'_>,
    metadata: &ModelMetadata<'_>,
    tokenizer: &dyn TokenizerTrait,
) -> Result<Vec<PreparedMultimodalPart>> {
    let preprocessed_parts = try_join_all(
        media_batches
            .iter()
            .map(|media| preprocess_modality(media, ctx)),
    )
    .await?;

    media_batches
        .into_iter()
        .zip(preprocessed_parts)
        .map(|(media, preprocessed)| prepared_part(media, preprocessed, ctx, metadata, tokenizer))
        .collect()
}

/// One batch's prepared part: its preprocessed inputs checked against its
/// item count, its prompt replacements, its placeholder tokens and the spec's
/// layouts for the modality.
fn prepared_part(
    media: MediaBatch,
    preprocessed: PreprocessedEncoderInputs,
    ctx: &PlanContext<'_>,
    metadata: &ModelMetadata<'_>,
    tokenizer: &dyn TokenizerTrait,
) -> Result<PreparedMultimodalPart> {
    let spec = ctx.spec;
    let modality = media.modality();
    debug!(
        ?modality,
        item_count = preprocessed.feature_token_counts.len(),
        total_tokens = preprocessed.feature_token_counts.iter().sum::<usize>(),
        "Multimodal preprocessing complete"
    );

    let media_info = media_item_infos(&media);
    // The engine never places an item above its encoder budget: refuse one
    // here, on the count the expansion below uses.
    check_item_feature_budget(
        modality,
        &preprocessed.feature_token_counts,
        &media_info,
        ctx.encoder_budget,
    )?;
    let prompt_replacements = spec
        .prompt_replacements_with_media(
            metadata,
            &preprocessed,
            modality,
            &media_info,
            preprocessor_config_for(ctx.model_config, modality),
        )
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
        .placeholder_token_for(metadata, modality)
        .map_err(|e| anyhow::anyhow!("Failed to get placeholder token: {e}"))?;
    let search_token_id = tokenizer.token_to_id(&placeholder_token);
    let placeholder_token_id: Option<u32> = match spec.placeholder_token_id_for(metadata, modality)
    {
        Ok(id) => Some(u32::try_from(id).map_err(|_| {
            anyhow::anyhow!("Invalid negative placeholder token ID {id} for modality {modality}")
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

    Ok(PreparedMultimodalPart {
        preprocessed,
        media,
        prompt_replacements,
        search_token_id,
        placeholder_token_id,
        field_layouts: spec.encoder_field_layouts_for(modality),
        keep_on_cpu_keys: spec.keep_on_cpu_keys_for(modality),
        encoder_input_key: spec.encoder_input_key_for(modality),
    })
}

/// The prompt expanded with every part's replacements, and the bindings of
/// each part's items to the expanded positions.
fn expand_prompt(
    token_ids: &[u32],
    prepared_parts: &[PreparedMultimodalPart],
) -> Result<ExpandedMultimodalTokens> {
    let expansions = prepared_parts
        .iter()
        .map(|part| ModalityExpansion {
            modality: part.media.modality(),
            search_token_id: part.search_token_id,
            placeholder_token_id: part.placeholder_token_id,
            replacements: &part.prompt_replacements,
        })
        .collect::<Vec<_>>();
    let expanded = expand_tokens_for_modalities(token_ids, &expansions)?;
    let placeholder_count = expanded.bindings.iter().map(Vec::len).sum::<usize>();

    debug!(
        original_len = token_ids.len(),
        expanded_len = expanded.token_ids.len(),
        placeholder_count,
        modality_count = prepared_parts.len(),
        "Token expansion complete"
    );
    Ok(expanded)
}

/// The lightweight intermediate of the prepared parts and their bindings;
/// tensor serialization is deferred to assembly.
fn intermediate_of(
    prepared_parts: Vec<PreparedMultimodalPart>,
    bindings: Vec<Vec<PromptBinding>>,
    spec: &dyn ModelProcessorSpec,
) -> Result<MultimodalIntermediate> {
    let batches = prepared_parts
        .into_iter()
        .zip(bindings)
        .map(|(part, bindings)| PrecomputedMultimodalIntermediate {
            preprocessed: part.preprocessed,
            media: part.media,
            bindings,
            placeholder_token_id: part.placeholder_token_id,
            field_layouts: part.field_layouts,
            keep_on_cpu_keys: part.keep_on_cpu_keys,
            encoder_input_key: part.encoder_input_key,
        })
        .collect::<Vec<_>>();
    Ok(MultimodalIntermediate::try_new(batches)?
        .with_encoder_input_format(spec.encoder_input_format()))
}

/// Preprocess one modality's batch: a small image batch through the pixel
/// cache, image by image, when the cache is on and the processor allows it;
/// every other batch whole, on the blocking pool.
async fn preprocess_modality(
    media: &MediaBatch,
    ctx: &PlanContext<'_>,
) -> Result<PreprocessedEncoderInputs> {
    if let Some(preprocessed) = cached_images(media, ctx).await? {
        return Ok(preprocessed);
    }
    preprocess_batch(media, ctx).await
}

/// The per-image path through the pixel cache for an image batch the cache
/// takes: one image, or fewer than 32 of a processor that preprocesses each
/// image independently (retaining individual tensors plus their concatenation
/// is costly for large requests, which keep their whole-batch allocation
/// path). `None` sends the batch down the whole-batch path.
async fn cached_images(
    media: &MediaBatch,
    ctx: &PlanContext<'_>,
) -> Result<Option<PreprocessedEncoderInputs>> {
    let MediaBatch::Images(images) = media else {
        return Ok(None);
    };
    let Some(cache) = &ctx.components.pixel_cache else {
        return Ok(None);
    };
    let processor = ctx.vision_processors.image();
    if !(images.len() == 1
        || (images.len() < 32 && processor.is_ok_and(|p| p.supports_per_image_preprocessing())))
    {
        return Ok(None);
    }
    let mut config = serde_json::json!({
        "model_id": ctx.model_id,
        "model_type": ctx.model_type,
        "model_config": ctx.model_config.config,
        "preprocessor_config": preprocessor_config_for(ctx.model_config, Modality::Image),
        // A one-pass result is only right for its own wire.
        "image_wire_dtype": ctx.image_wire_dtype.map(|dtype| dtype.to_string()),
        "transparency": ctx.transparency,
    });
    config.sort_all_objects();
    let fingerprint = config_fingerprint(ctx.tokenizer_id, &config);
    // Deduplicate within the request even if the result cannot fit
    // in the cache. Record every position before scheduling work.
    let mut unique = HashMap::new();
    let mut positions: Vec<Vec<usize>> = Vec::new();
    let tasks = images
        .iter()
        .enumerate()
        .filter_map(|(position, image)| {
            let next = unique.len();
            let index = *unique
                .entry(PixelCacheKey {
                    image_hash: image.hash.clone(),
                    config_fingerprint: fingerprint,
                })
                .or_insert(next);
            if index == next {
                positions.push(vec![position]);
                Some(image)
            } else {
                positions[index].push(position);
                None
            }
        })
        .map(|image| {
            preprocess_image_cached(
                cache.clone(),
                image.clone(),
                ctx.vision_processors.clone(),
                ctx.model_id.to_string(),
                fingerprint,
                ctx.image_wire_dtype,
            )
        })
        .collect::<Vec<_>>();
    let parts = futures::stream::iter(tasks)
        // Keep unique results aligned with their recorded positions.
        .buffered(4)
        .try_collect::<Vec<_>>()
        .await?;
    let layouts = ctx
        .spec
        .encoder_field_layouts_for(Modality::Image)
        .model_specific;
    let image_count = images.len();
    tokio::task::spawn_blocking(move || -> Result<_> {
        let mut ordered = vec![None; image_count];
        for (part, positions) in parts.into_iter().zip(positions) {
            let (&last, repeats) = positions
                .split_last()
                .context("Unique image has no position")?;
            for &position in repeats {
                ordered[position] = Some(part.clone());
            }
            ordered[last] = Some(part);
        }
        let parts = ordered
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .context("Missing image output position")?;
        PreprocessedEncoderInputs::concat(parts, &layouts)
    })
    .await
    .context("Image batch assembly task panicked")?
    .context("Image batch assembly failed")
    .map(Some)
}

/// The whole batch preprocessed on the blocking pool: the CPU-intensive
/// preprocessing must not block the tokio async runtime under concurrent
/// load. TODO: consider making the thread pool size configurable.
async fn preprocess_batch(
    media: &MediaBatch,
    ctx: &PlanContext<'_>,
) -> Result<PreprocessedEncoderInputs> {
    let vision_processors = ctx.vision_processors.clone();
    let media_for_preprocess = media.clone(); // cheap Arc refcount bumps
    let video_layouts = ctx
        .spec
        .encoder_field_layouts_for(Modality::Video)
        .model_specific;
    let audio_processor = if media.modality() == Modality::Audio {
        Some(
            ctx.spec
                .audio_processor(
                    &ctx.model_config.config,
                    &ctx.model_config.preprocessor_config,
                )
                .map_err(|error| {
                    anyhow::anyhow!(
                        "The audio processor of model spec {} cannot be built: {error}",
                        ctx.spec.name()
                    )
                })?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No audio processor registered for model spec: {}",
                        ctx.spec.name()
                    )
                })?,
        )
    } else {
        None
    };
    let image_wire_dtype = ctx.image_wire_dtype;

    tokio::task::spawn_blocking(move || match media_for_preprocess {
        MediaBatch::Images(images) => {
            // Decode every frame first, in parallel on the preprocessing
            // pool, rather than one photo after another on this thread.
            ImageFrame::decode_all(&images)
                .map_err(llm_multimodal::MultiModalError::Media)?;
            let processor = vision_processors.image()?;
            // Decoded (and cloned) inside the blocking closure, off the async
            // runtime.
            let raw_images: Vec<image::DynamicImage> = images
                .iter()
                .map(|frame| {
                    frame
                        .image()
                        .cloned()
                        .map_err(llm_multimodal::MultiModalError::Media)
                })
                .collect::<Result<_, _>>()?;
            preprocess_images(processor.as_ref(), &raw_images, image_wire_dtype)
        }
        MediaBatch::Videos(videos) => {
            let processor = vision_processors.video()?;
            // The processor takes one decoded clip at a time; the clips are
            // joined into one batch afterwards, laid out as if processed together.
            let preprocess_clip = |video: &VideoClip| -> Result<PreprocessedEncoderInputs> {
                let clip = video_clip_options(video);

                if !video.frames().is_empty() {
                    return processor
                        .preprocess_video(video.frames(), &clip)
                        .map_err(|e| anyhow::anyhow!("Video preprocessing failed: {e}"));
                }

                if let Some(rgb_video) = video.rgb_video() {
                    match rgb_video.frame_refs() {
                        Ok(frame_refs) => match processor.preprocess_video_rgb(&frame_refs, &clip) {
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
                    .preprocess_video(&frames, &clip)
                    .map_err(|e| anyhow::anyhow!("Video preprocessing failed: {e}"))
            };

            // Clips are independent until the concat below, and each one is
            // frames' worth of work, so they go wide the way the image batch
            // does. `collect` keeps request order, which the concat relies on.
            let parts = videos
                .par_iter()
                .map(|video| preprocess_clip(video))
                .collect::<Result<Vec<_>>>()?;
            PreprocessedEncoderInputs::concat(parts, &video_layouts)
                .map_err(|e| anyhow::anyhow!("Video batch assembly failed: {e}"))
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
    .map_err(|e| anyhow::Error::new(e).context("Preprocessing task panicked"))?
}

/// What the processor reads about one clip besides its frames: its sampled
/// frame rate and, when the caller named a `max_long_side_pixel` tier, that
/// tier, so the processor can size the frames the way the caller asked.
fn video_clip_options(video: &VideoClip) -> VideoClipOptions {
    VideoClipOptions {
        sample_fps: video.sample_fps(),
        max_long_side_pixel: video.max_long_side_pixel(),
    }
}

/// The config a modality is preprocessed with: video's own when the checkpoint ships one.
pub(super) fn preprocessor_config_for(
    model_config: &MultimodalModelConfig,
    modality: Modality,
) -> &PreProcessorConfig {
    match modality {
        Modality::Video => model_config
            .video_preprocessor_config
            .as_ref()
            .unwrap_or(&model_config.preprocessor_config),
        _ => &model_config.preprocessor_config,
    }
}

/// One descriptor per media item, in batch order; only decoded clips carry sampling.
fn media_item_infos(media: &MediaBatch) -> Vec<MediaItemInfo> {
    match media {
        MediaBatch::Videos(videos) => videos
            .iter()
            .map(|clip| MediaItemInfo {
                video_sampling: clip.sampling().cloned(),
            })
            .collect(),
        MediaBatch::Images(_) | MediaBatch::Audios(_) => {
            vec![MediaItemInfo::default(); media.len()]
        }
    }
}

/// Pixel-cache lookup and preprocessing for one image.
async fn preprocess_image_cached(
    cache: Arc<PixelCache>,
    image: Arc<ImageFrame>,
    vision_processors: Arc<VisionProcessors>,
    model_id: String,
    fingerprint: u64,
    image_wire_dtype: Option<EncoderDtype>,
) -> Result<PreprocessedEncoderInputs> {
    let key = PixelCacheKey {
        image_hash: image.hash.clone(),
        config_fingerprint: fingerprint,
    };
    if let Some(cached) = cache.get(&key) {
        Metrics::record_mm_pixel_cache_lookup(&model_id, true);
        return Ok(cached.preprocessed.clone());
    }
    Metrics::record_mm_pixel_cache_lookup(&model_id, false);

    let preprocessed = preprocess_image_batch(
        vision_processors,
        std::slice::from_ref(&image),
        image_wire_dtype,
    )
    .await?;
    // The entry is the form the request is sent in: the wire bytes when a
    // dtype was resolved (twice the neutral bytes at the half widths, four
    // times at float32, against the same budget), so a hit skips the write
    // pass; the neutral patches otherwise.
    cache.insert(
        key,
        Arc::new(CachedPreprocessedItem {
            preprocessed: preprocessed.clone(),
        }),
    );
    Ok(preprocessed)
}

async fn preprocess_image_batch(
    vision_processors: Arc<VisionProcessors>,
    images: &[Arc<ImageFrame>],
    image_wire_dtype: Option<EncoderDtype>,
) -> Result<PreprocessedEncoderInputs> {
    let images = images.to_vec();
    tokio::task::spawn_blocking(move || {
        let raw_images: Vec<image::DynamicImage> = images
            .iter()
            .map(|frame| {
                frame
                    .image()
                    .cloned()
                    .map_err(llm_multimodal::MultiModalError::Media)
            })
            .collect::<Result<_, _>>()?;
        let processor = vision_processors.image()?;
        preprocess_images(processor.as_ref(), &raw_images, image_wire_dtype)
    })
    .await
    .map_err(|e| anyhow::Error::new(e).context("Preprocessing task panicked"))?
}

/// The images preprocessed: laid out in `wire_dtype` in one pass when the
/// dtype is known and the processor writes it, through the per-destination
/// write otherwise. Both arms produce the same bytes on the wire.
fn preprocess_images(
    processor: &dyn VisionPreProcessor,
    images: &[image::DynamicImage],
    wire_dtype: Option<EncoderDtype>,
) -> Result<PreprocessedEncoderInputs> {
    wire_dtype
        .and_then(|dtype| processor.preprocess_wire(images, dtype))
        .unwrap_or_else(|| processor.preprocess(images))
        .map_err(|e| anyhow::anyhow!("Image preprocessing failed: {e}"))
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
    use llm_multimodal::{ImageDetail, ImageSource, VideoSamplingInfo, VideoSource};

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

        let clip = video_clip_options(&video);

        assert_eq!(clip.sample_fps, 0.8);
        assert!(clip.max_long_side_pixel.is_none());
    }

    #[test]
    fn decoded_video_tier_reaches_the_processor() {
        let video = VideoClip::new_with_sample_fps(
            Vec::new(),
            Bytes::new(),
            VideoSource::InlineBytes,
            "video-hash".to_string(),
            2.0,
        )
        .with_max_long_side_pixel(Some(1008));

        let clip = video_clip_options(&video);

        assert_eq!(clip.max_long_side_pixel, Some(1008));
    }

    #[test]
    fn media_item_infos_carry_video_sampling_per_clip() {
        let sampling = VideoSamplingInfo {
            source_fps: 30.0,
            frame_indices: vec![0, 15, 30],
        };
        let sampled = VideoClip::new(
            Vec::new(),
            Bytes::new(),
            VideoSource::InlineBytes,
            "sampled".to_string(),
        )
        .with_sampling(Some(sampling.clone()));
        let unsampled = VideoClip::new(
            Vec::new(),
            Bytes::new(),
            VideoSource::InlineBytes,
            "unsampled".to_string(),
        );

        let infos = media_item_infos(&MediaBatch::Videos(vec![
            Arc::new(sampled),
            Arc::new(unsampled),
        ]));

        assert_eq!(
            infos,
            vec![
                MediaItemInfo {
                    video_sampling: Some(sampling)
                },
                MediaItemInfo::default()
            ]
        );
    }

    #[test]
    fn media_item_infos_default_for_every_image() {
        let image = || {
            Arc::new(ImageFrame::new(
                image::DynamicImage::new_rgb8(1, 1),
                Bytes::new(),
                ImageDetail::Auto,
                ImageSource::InlineBytes,
                "hash".to_string(),
            ))
        };

        let infos = media_item_infos(&MediaBatch::Images(vec![image(), image(), image()]));

        assert_eq!(infos, vec![MediaItemInfo::default(); 3]);
    }

    #[test]
    fn preprocessor_config_for_video_is_the_video_config_when_shipped() {
        let image_config = PreProcessorConfig {
            temporal_patch_size: Some(2),
            ..Default::default()
        };
        let video_config = PreProcessorConfig {
            temporal_patch_size: Some(4),
            ..Default::default()
        };
        let with_video = MultimodalModelConfig::new(
            serde_json::json!({}),
            image_config.clone(),
            Some(video_config),
        );
        let without_video = MultimodalModelConfig::new(serde_json::json!({}), image_config, None);

        let temporal = |config: &MultimodalModelConfig, modality| {
            preprocessor_config_for(config, modality).temporal_patch_size
        };
        assert_eq!(temporal(&with_video, Modality::Video), Some(4));
        assert_eq!(temporal(&with_video, Modality::Image), Some(2));
        assert_eq!(temporal(&with_video, Modality::Audio), Some(2));
        assert_eq!(temporal(&without_video, Modality::Video), Some(2));
    }

    #[test]
    fn test_explicit_feature_ranges_keep_non_uniform_strides() {
        let replacements =
            vec![
                PromptReplacement::sequence(Modality::Video, "<video>", vec![7; 48])
                    .with_feature_ranges(vec![
                        PlaceholderRange {
                            offset: 5,
                            length: 4,
                        },
                        PlaceholderRange {
                            offset: 19,
                            length: 4,
                        },
                        PlaceholderRange {
                            offset: 40,
                            length: 4,
                        },
                    ]),
            ];
        let expansion = ModalityExpansion {
            modality: Modality::Video,
            search_token_id: Some(100),
            placeholder_token_id: Some(7),
            replacements: &replacements,
        };

        let result = expand_tokens_for_modalities(&[1, 2, 100, 3], &[expansion]).unwrap();

        assert_eq!(result.token_ids.len(), 3 + 48);
        assert_eq!(result.bindings[0][0].structural.offset, 2);
        assert_eq!(result.bindings[0][0].structural.length, 48);
        assert_eq!(
            result.bindings[0][0].patches,
            vec![
                PlaceholderRange {
                    offset: 7,
                    length: 4,
                },
                PlaceholderRange {
                    offset: 21,
                    length: 4,
                },
                PlaceholderRange {
                    offset: 42,
                    length: 4,
                },
            ]
        );
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

#[cfg(test)]
#[path = "image_cache_tests.rs"]
mod image_cache_tests;
