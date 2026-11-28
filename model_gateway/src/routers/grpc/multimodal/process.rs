//! Multimodal processing core: fetch media → preprocess pixels → expand
//! placeholder tokens → build the lightweight [`MultimodalIntermediate`].
//!
//! Protocol adapters first normalize media into a [`MediaPlan`], so this module
//! has one request-independent processing entry point.

use std::{collections::HashMap, sync::Arc, time::Instant};

use anyhow::{Context, Result};
use futures::{future::try_join_all, StreamExt, TryStreamExt};
use llm_multimodal::{
    AsyncMultiModalTracker, AudioClip, EncoderDtype, EncoderFieldLayouts, FrameSampling,
    ImageFrame, MediaItemInfo, Modality, ModelMetadata, ModelProcessorSpec, PlaceholderRange,
    PreProcessorConfig, PreprocessedEncoderInputs, PromptReplacement, TrackedMedia, TrackerOutput,
    TransformError, TransparencyPolicy, VideoClip, VideoClipOptions, VisionPreProcessor,
};
use llm_tokenizer::TokenizerTrait;
use rayon::prelude::*;
use tracing::{debug, info, warn};

use super::{
    config::{MultimodalComponents, MultimodalModelConfig, VisionProcessors},
    encoder_budget::{
        check_image_size_feature_budget, check_image_wire_cap, check_item_feature_budget,
    },
    log_mm_timing_enabled,
    pixel_cache::{config_fingerprint, CachedPreprocessedItem, PixelCache, PixelCacheKey},
    plan::MediaPlan,
    transparency::frame_transparency,
    MediaBatch, MultimodalIntermediate, MultimodalOutput, PrecomputedMultimodalIntermediate,
    PromptBinding, RegistryTokenizer,
};
use crate::observability::metrics::Metrics;

#[path = "batch_image_cache.rs"]
mod batch_image_cache;

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
    /// The byte bound the images' pixel tensors must fit before any is
    /// decoded, when the caller knows they ride inline in the gateway's own
    /// gRPC message (see [`check_image_wire_cap`]); `None` where they travel
    /// beside it, and the framing of that hop answers.
    image_wire_cap: Option<usize>,
    transparency: TransparencyPolicy,
    /// The engine's per-item bound in feature tokens, when known (see
    /// [`super::encoder_budget`]): held to on an image's header size before
    /// anything is decoded, and on every item's produced count after its
    /// preprocessing.
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

/// The refusal of an image whose size the processor's own size rule
/// refuses, found on its header before any decode: the request's own media,
/// refused as sent on both pipelines (a 400 under the media stage's code;
/// from a worker, a media fault of the client's).
#[derive(Debug, thiserror::Error)]
#[error("Image preprocessing failed: {0}")]
pub(crate) struct ImageSizeRefused(pub(crate) String);

impl ImageSizeRefused {
    /// Whether `error` is this refusal, at any depth of its chain.
    pub(crate) fn is(error: &anyhow::Error) -> bool {
        error
            .chain()
            .any(|cause| cause.downcast_ref::<Self>().is_some())
    }
}

/// The refusal of a video part by a process that cannot decode video
/// (`llm_multimodal::missing_video_decoder`): the request's own answer, a
/// 400 under the media stage's code like a modality the model does not
/// take, known at start-up and never a server error per request.
#[derive(Debug)]
pub(crate) struct VideoDecoderUnavailable {
    pub(crate) why: &'static str,
}

impl VideoDecoderUnavailable {
    /// Whether `error` is this refusal, at any depth of its chain.
    pub(crate) fn is(error: &anyhow::Error) -> bool {
        error
            .chain()
            .any(|cause| cause.downcast_ref::<Self>().is_some())
    }
}

impl std::fmt::Display for VideoDecoderUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "video input cannot be decoded by this process: {}",
            self.why
        )
    }
}

impl std::error::Error for VideoDecoderUnavailable {}

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
/// known (see [`super::encoder_budget`]): an image above it on its header
/// size alone is refused before any of its batch is decoded or any tensor
/// built, and any item above it on its produced count is refused once its
/// features are counted; nothing above it reaches the engine.
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
    image_wire_cap: Option<usize>,
    transparency: TransparencyPolicy,
    encoder_budget: Option<usize>,
) -> Result<MultimodalOutput> {
    // A video part needs a decoder this process may lack, known at
    // start-up: refused up front, before any fetch or decode.
    if plan.count(Modality::Video) > 0 {
        if let Some(why) = llm_multimodal::missing_video_decoder() {
            return Err(VideoDecoderUnavailable { why }.into());
        }
    }
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
        image_wire_cap,
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

/// The spec's view of a clip for this model: its sampling rate and frame
/// placement, read from the config its video processor is built from (see
/// [`preprocessor_config_for`]) so the frames the connector picks and the
/// frames the processor groups answer one config whatever files the
/// checkpoint ships; the placement under the engine's frame budget. A rate
/// the request states (`request_fps`) goes ahead of the rate the engine's
/// start states (`start_fps`), and that ahead of the config's: it is the
/// default of the plan's clips, and a part's own rate still wins.
fn video_sampling(
    spec: &dyn ModelProcessorSpec,
    model_config: &MultimodalModelConfig,
    frame_budget: Option<usize>,
    start_fps: Option<f64>,
    request_fps: Option<f64>,
) -> (Option<f64>, FrameSampling) {
    let config = preprocessor_config_for(model_config, Modality::Video);
    (
        request_fps
            .or(start_fps)
            .or_else(|| spec.default_video_sample_fps_with(config)),
        spec.video_frame_sampling_with(config)
            .with_frame_budget(frame_budget),
    )
}

/// Fetch and decode the plan's media through the tracker, under the spec's
/// video sampling and the frames' transparency policy.
async fn fetch_media(plan: MediaPlan, ctx: &PlanContext<'_>) -> Result<Fetched> {
    let (sample_fps, frame_sampling) = video_sampling(
        ctx.spec,
        ctx.model_config,
        ctx.components.video_frame_budget,
        ctx.components.video_start_fps,
        plan.request_video_fps(),
    );
    let mut tracker = AsyncMultiModalTracker::new(ctx.components.media_connector.clone())
        .with_drain_on_error(ctx.components.drain_fetches_on_error)
        .with_fetch_observer(
            Metrics::interned_model_label(ctx.model_id),
            Metrics::record_mm_fetch,
        )
        .with_default_video_sample_fps(sample_fps)
        .with_video_frame_sampling(frame_sampling)
        .with_image_transparency(ctx.transparency)
        .with_minimax_sglang_resize(
            ctx.spec.resizes_source_images()
                && ctx
                    .model_config
                    .preprocessor_config
                    .get_extra::<bool>(llm_multimodal::minimax_resize::SGLANG_RESIZE_KEY)
                    .unwrap_or(false),
        )
        .with_minimax_image_resize(
            ctx.spec.resizes_source_images()
                && !ctx
                    .model_config
                    .preprocessor_config
                    .get_extra::<bool>(llm_multimodal::minimax_resize::SGLANG_RESIZE_KEY)
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
    let image_prompt_tokens = image_prompt_tokens(
        prepared_parts
            .iter()
            .map(|part| (part.media.modality(), part.prompt_replacements.as_slice())),
    );
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
        .with_encoder_input_format(spec.encoder_input_format())
        .with_image_prompt_tokens(image_prompt_tokens)
        .with_language_model_reads_grids(spec.language_model_reads_grids()))
}

/// The tokens the expansion writes into the prompt for the request's images:
/// the image part's replacements summed, each the whole run that stands in
/// for one image anchor and that the engine counts as prompt tokens; what the
/// chat usage reports as `prompt_tokens_details.image_tokens`. `None` when no
/// part is images: the other modalities' placeholders are not image tokens.
fn image_prompt_tokens<'a>(
    parts: impl IntoIterator<Item = (Modality, &'a [PromptReplacement])>,
) -> Option<u32> {
    parts
        .into_iter()
        .find(|(modality, _)| *modality == Modality::Image)
        .map(|(_, replacements)| {
            replacements.iter().fold(0u32, |tokens, replacement| {
                tokens.saturating_add(u32::try_from(replacement.tokens.len()).unwrap_or(u32::MAX))
            })
        })
}

/// Preprocess one modality's batch, using the pixel cache for eligible images.
/// An image the engine can
/// never place is refused first, on the size its header gave and the
/// processor's own size-to-count rule, before any of the batch is decoded
/// or any tensor built; the counts the preprocessing produces are held to
/// the same budget in [`prepared_part`]. A batch whose pixel tensors could
/// not ride in one message to the engine is refused at the same point, on
/// the same sizes, as the request's own, where the caller knows they ride
/// inline in the gateway's own message ([`check_image_wire_cap`]). A size
/// the processor's rule refuses is refused there too, in the preprocessing's
/// words ([`ImageSizeRefused`]).
async fn preprocess_modality(
    media: &MediaBatch,
    ctx: &PlanContext<'_>,
) -> Result<PreprocessedEncoderInputs> {
    if let MediaBatch::Images(images) = media {
        if let Ok(processor) = ctx.vision_processors.image() {
            let refused =
                check_image_size_feature_budget(processor.as_ref(), images, ctx.encoder_budget)?;
            check_image_wire_cap(
                processor.as_ref(),
                images,
                ctx.image_wire_dtype,
                ctx.image_wire_cap,
            )
            .map_err(llm_multimodal::MultiModalError::Media)?;
            if let Some(refused) = refused {
                return Err(ImageSizeRefused(refused.to_string()).into());
            }
        }
    }
    if let Some(preprocessed) = cached_images(media, ctx).await? {
        return Ok(preprocessed);
    }
    preprocess_batch(media, ctx).await
}

/// Batch misses for processors that support slicing the result into cache
/// entries. Use the per-image path for other eligible requests.
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
    let independent = processor
        .as_ref()
        .is_ok_and(|processor| processor.supports_per_image_preprocessing());
    let batch_cache = ctx.image_wire_dtype == Some(EncoderDtype::BFloat16)
        && !images.is_empty()
        && processor.is_ok_and(|processor| processor.supports_bf16_batch_cache());
    let per_image_cache = images.len() == 1 || (images.len() < 32 && independent);
    if batch_cache {
        return batch_image_cache::preprocess(images, cache, ctx, !per_image_cache).await;
    }
    if !per_image_cache {
        return Ok(None);
    }
    let fingerprint = image_cache_fingerprint(ctx)?;
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
    on_pool(ctx.model_id, move || -> Result<_> {
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
        PreprocessedEncoderInputs::concat_refs(
            &parts
                .iter()
                .map(|part| &part.preprocessed)
                .collect::<Vec<_>>(),
            &layouts,
        )
    })
    .await
    .context("Image batch assembly task panicked")?
    .context("Image batch assembly failed")
    .map(Some)
}

/// Identify the model, wire format and image policy in the pixel cache:
/// hashed by the first request of a loaded config with this model, wire
/// and policy, read back by every later one.
fn image_cache_fingerprint(ctx: &PlanContext<'_>) -> Result<u64> {
    ctx.model_config.image_fingerprint(
        ctx.tokenizer_id,
        ctx.model_id,
        ctx.image_wire_dtype,
        ctx.transparency,
        || {
            let config = serde_json::json!({
                "model_id": ctx.model_id,
                "model_type": ctx.model_type,
                "preprocessor_config": preprocessor_config_for(ctx.model_config, Modality::Image),
                // A one-pass result is only right for its own wire.
                "image_wire_dtype": ctx.image_wire_dtype.map(|dtype| dtype.to_string()),
                "transparency": ctx.transparency,
            });
            Ok(config_fingerprint(
                ctx.tokenizer_id,
                &config,
                &ctx.model_config.config,
            )?)
        },
    )
}

/// `work` on the blocking pool, with the time its task waited in the pool's
/// queue (the gap between the spawn and the closure's first instruction)
/// recorded as the `pool_queue` stage of `model_id`: one observation per
/// task, so a burst that queues behind the pool's threads is seen as waiting,
/// apart from the work itself.
async fn on_pool<T: Send + 'static>(
    model_id: &str,
    work: impl FnOnce() -> T + Send + 'static,
) -> std::result::Result<T, tokio::task::JoinError> {
    let model = Metrics::interned_model_label(model_id);
    on_pool_recording(work, move |waited| {
        Metrics::record_mm_processing_stage(&model, "pool_queue", waited);
    })
    .await
}

/// `work` on the blocking pool; `record` takes the seconds its task waited
/// for a pool thread, measured at the closure's first instruction on that
/// thread. The record happens there and not on the awaiting side, so a task
/// whose request was dropped while it queued, or whose work panics, still
/// records its wait: under a burst those are the long tail this stage is
/// for.
async fn on_pool_recording<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
    record: impl FnOnce(f64) + Send + 'static,
) -> std::result::Result<T, tokio::task::JoinError> {
    let queued_at = Instant::now();
    tokio::task::spawn_blocking(move || {
        record(queued_at.elapsed().as_secs_f64());
        work()
    })
    .await
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
        Some(ctx.model_config.audio_processor(ctx.spec)?)
    } else {
        None
    };
    let image_wire_dtype = ctx.image_wire_dtype;

    on_pool(ctx.model_id, move || match media_for_preprocess {
        MediaBatch::Images(images) => {
            preprocess_image_frames(&images, &vision_processors, image_wire_dtype)
        }
        MediaBatch::Videos(videos) => {
            let processor = vision_processors.video()?;
            // The processor takes one decoded clip at a time; the clips are
            // joined into one batch afterwards, laid out as if processed together.

            // Clips are independent until the concat below, and each one is
            // frames' worth of work, so they go wide the way the image batch
            // does. `collect` keeps request order, which the concat relies on.
            let parts = videos
                .par_iter()
                .map(|video| preprocess_clip(processor.as_ref(), video))
                .collect::<Result<Vec<_>>>()?;
            PreprocessedEncoderInputs::concat(parts, &video_layouts)
                .map_err(|e| anyhow::anyhow!("Video batch assembly failed: {e}"))
        }
        MediaBatch::Audios(audios) => {
            let processor = audio_processor
                .ok_or_else(|| anyhow::anyhow!("Model did not provide an audio processor"))?;
            processor
                .preprocess(&audios)
                .map_err(|e| anyhow::anyhow!("Audio preprocessing failed: {e}"))
        }
    })
    .await
    .map_err(|e| anyhow::Error::new(e).context("Preprocessing task panicked"))?
}

/// One decoded clip through the processor: the frames it holds, else its RGB
/// buffers by the fast path, else the frames materialized from them.
fn preprocess_clip(
    processor: &dyn VisionPreProcessor,
    video: &VideoClip,
) -> Result<PreprocessedEncoderInputs> {
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
                // The frame path would ask for the same output again.
                Err(error @ TransformError::ReserveRefused(_)) => {
                    return Err(anyhow::anyhow!("Video preprocessing failed: {error}"));
                }
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
) -> Result<Arc<CachedPreprocessedItem>> {
    let key = PixelCacheKey {
        image_hash: image.hash.clone(),
        config_fingerprint: fingerprint,
    };
    if let Some(cached) = cache.get(&key) {
        Metrics::record_mm_pixel_cache_lookup(&model_id, true);
        return Ok(cached);
    }
    Metrics::record_mm_pixel_cache_lookup(&model_id, false);

    let preprocessed = preprocess_image_batch(
        &model_id,
        vision_processors,
        std::slice::from_ref(&image),
        image_wire_dtype,
    )
    .await?;
    // The entry is the form the request is sent in: the wire bytes when a
    // dtype was resolved (twice the neutral bytes at the half widths, four
    // times at float32, against the same budget), so a hit skips the write
    // pass; the neutral patches otherwise.
    let cached = Arc::new(CachedPreprocessedItem { preprocessed });
    cache.insert(key, cached.clone());
    Ok(cached)
}

async fn preprocess_image_batch(
    model_id: &str,
    vision_processors: Arc<VisionProcessors>,
    images: &[Arc<ImageFrame>],
    image_wire_dtype: Option<EncoderDtype>,
) -> Result<PreprocessedEncoderInputs> {
    let images = images.to_vec();
    on_pool(model_id, move || {
        // Borrowed from the frames' own cells, as the batch path borrows them.
        let decoded: Vec<&image::DynamicImage> = images
            .iter()
            .map(|frame| {
                frame
                    .image()
                    .map_err(llm_multimodal::MultiModalError::Media)
            })
            .collect::<Result<_, _>>()?;
        let processor = vision_processors.image()?;
        preprocess_images(processor.as_ref(), &decoded, image_wire_dtype)
    })
    .await
    .map_err(|e| anyhow::Error::new(e).context("Preprocessing task panicked"))?
}

/// Decode a batch in parallel and borrow its frames for the vision processor.
fn preprocess_image_frames(
    images: &[Arc<ImageFrame>],
    vision_processors: &VisionProcessors,
    image_wire_dtype: Option<EncoderDtype>,
) -> Result<PreprocessedEncoderInputs> {
    // Decode every frame first, in parallel on the preprocessing
    // pool, rather than one photo after another on this thread.
    ImageFrame::decode_all(images).map_err(llm_multimodal::MultiModalError::Media)?;
    let processor = vision_processors.image()?;
    // The frames' own pixels, borrowed from the cells `decode_all`
    // just filled: no processor owns, consumes or mutates its input,
    // so the batch costs one pointer per image instead of a deep copy
    // of every decoded image.
    let decoded: Vec<&image::DynamicImage> = images
        .iter()
        .map(|frame| {
            frame
                .image()
                .map_err(llm_multimodal::MultiModalError::Media)
        })
        .collect::<Result<_, _>>()?;
    preprocess_images(processor.as_ref(), &decoded, image_wire_dtype)
}

/// The images preprocessed: laid out in `wire_dtype` in one pass when the
/// dtype is known and the processor writes it, through the per-destination
/// write otherwise. Both arms produce the same bytes on the wire.
fn preprocess_images(
    processor: &dyn VisionPreProcessor,
    images: &[&image::DynamicImage],
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
#[path = "../../../../tests/unit/routers/grpc/multimodal/process/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/process/image_cache_tests.rs"]
mod image_cache_tests;
