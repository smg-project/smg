//! Cache images independently while preprocessing misses as a batch.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{Context, Result};
use llm_multimodal::{
    EncoderDtype, EncoderFieldLayouts, EncoderInput, FieldLayout, ImageFrame, Modality,
    PreprocessedEncoderInputs,
};

use super::{image_cache_fingerprint, on_pool, preprocess_images, PlanContext, VisionProcessors};
use crate::{
    observability::metrics::Metrics,
    routers::grpc::multimodal::pixel_cache::{CachedPreprocessedItem, PixelCache, PixelCacheKey},
};

/// Reuse individual images, batch the unique misses, and restore request order.
pub(super) async fn preprocess(
    images: &[Arc<ImageFrame>],
    cache: &Arc<PixelCache>,
    ctx: &PlanContext<'_>,
) -> Result<Option<PreprocessedEncoderInputs>> {
    let model_id = Metrics::interned_model_label(ctx.model_id);
    let wire_dtype = ctx.image_wire_dtype;
    let bf16 = wire_dtype == Some(EncoderDtype::BFloat16);
    let budget = cache.max_bytes();
    let width = if bf16 { 2 } else { 1 };
    let processor = ctx.vision_processors.image()?;
    let remaining = images
        .iter()
        .filter_map(|image| {
            let size = image.size();
            processor.image_input_elements(size.width, size.height)
        })
        .try_fold(budget / width, usize::checked_sub);
    if remaining.is_none()
        && images
            .iter()
            .map(|image| image.hash.as_str())
            .collect::<HashSet<_>>()
            .len()
            == images.len()
    {
        Metrics::record_mm_pixel_cache_budget_bypass(&model_id);
        return Ok(None);
    }
    let fingerprint = image_cache_fingerprint(ctx)?;
    let working_bytes = remaining.map_or(0, |remaining| (budget / width - remaining) * width);
    let images = images.to_vec();
    let cache = Arc::clone(cache);
    let processors = Arc::clone(&ctx.vision_processors);
    let layouts = ctx.spec.encoder_field_layouts_for(Modality::Image);
    on_pool(ctx.model_id, move || {
        let mut indices = HashMap::new();
        let mut order = Vec::with_capacity(images.len());
        let mut unique = Vec::new();
        for image in &images {
            let next = unique.len();
            let index = *indices.entry(image.hash.as_str()).or_insert(next);
            order.push(index);
            if index == next {
                unique.push(image);
            }
        }

        let distinct = unique.len() == images.len();

        let mut parts = Vec::new();
        let mut misses = Vec::new();
        let mut miss_images = Vec::new();
        for (index, image) in unique.into_iter().enumerate() {
            let key = PixelCacheKey {
                image_hash: image.hash.clone(),
                config_fingerprint: fingerprint,
            };
            let entry = cache.get(&key);
            Metrics::record_mm_pixel_cache_lookup(&model_id, entry.is_some());
            if entry.is_none() {
                misses.push((index, key));
                miss_images.push(Arc::clone(image));
            }
            parts.push(entry);
        }

        let has_misses = !misses.is_empty();
        let mut inserts = Vec::new();
        let mut direct_output = None;
        if has_misses {
            let direct = misses.len() == images.len();
            // For a large, mostly new request, keep one contiguous output.
            // Recompute the small set of hits rather than gather a second
            // full tensor. Keep the hit-assisted path if decoding fails.
            const MIN_RECOMPUTE_BYTES: usize = 8 * 1024 * 1024;
            if bf16
                && distinct
                && !direct
                && misses.len() >= images.len() - misses.len()
                && working_bytes >= MIN_RECOMPUTE_BYTES
            {
                if let Ok(mut batch) = preprocess_image_frames(&images, &processors, wire_dtype) {
                    if admit_shared_batch(
                        &mut batch,
                        &images,
                        &layouts,
                        &cache,
                        budget,
                        fingerprint,
                    )? {
                        return Ok(Some(batch));
                    }
                }
            }
            let processor = processors.image()?;
            let decoded = decode_image_frames(&miss_images)?;
            let items = if let Some(items) = (!direct && !bf16)
                .then(|| processor.preprocess_image_items(&decoded))
                .flatten()
            {
                items?
            } else {
                let mut batch = preprocess_images(processor.as_ref(), &decoded, wire_dtype)?;
                // The fused batch processor emits neutral pixels for a lone
                // image. Match the representation of multi-image entries.
                if bf16 && !matches!(batch.encoder_input, EncoderInput::Wire { .. }) {
                    let input = batch.encoder_input;
                    let shape = input.shape().to_vec();
                    let norm = input
                        .pixel_norm()
                        .context("image input has no pixel normalization")?;
                    let bytes = input
                        .into_bytes_as(EncoderDtype::BFloat16)
                        .map_err(anyhow::Error::msg)?;
                    batch.encoder_input =
                        EncoderInput::wire(bytes, EncoderDtype::BFloat16, shape, norm)
                            .map_err(anyhow::Error::msg)?;
                }
                if direct && batch.encoder_input.nbytes() > budget {
                    return Ok(Some(batch));
                }
                if bf16 {
                    batch.encoder_input.share_wire(budget);
                }
                let items = split_batch(&batch, &layouts, misses.len())?;
                if direct {
                    direct_output = Some(batch);
                }
                items
            };
            anyhow::ensure!(
                items.len() == misses.len()
                    && items
                        .iter()
                        .all(|item| item.batch_size() == 1 && item.feature_token_counts.len() == 1),
                "image preprocessing returned an unexpected item count"
            );
            let admit = items
                .iter()
                .chain(parts.iter().flatten().map(|part| &part.preprocessed))
                .map(|item| item.encoder_input.nbytes())
                .try_fold(budget, usize::checked_sub)
                .is_some();
            for ((index, key), preprocessed) in misses.into_iter().zip(items) {
                let entry = Arc::new(CachedPreprocessedItem { preprocessed });
                if admit {
                    inserts.push((key, Arc::clone(&entry)));
                }
                parts[index] = Some(entry);
            }
        }

        let output = if let Some(batch) = direct_output {
            batch
        } else {
            let ordered = order
                .into_iter()
                .map(|index| {
                    parts[index]
                        .as_ref()
                        .map(|entry| &entry.preprocessed)
                        .context("missing image cache output")
                })
                .collect::<Result<Vec<_>>>()?;
            let mut output =
                PreprocessedEncoderInputs::concat_refs(&ordered, &layouts.model_specific)?;
            // Repoint the entries of a growing request to its assembled output
            // so its next replay can reuse it and old backing groups can leave.
            if bf16
                && distinct
                && has_misses
                && admit_shared_batch(&mut output, &images, &layouts, &cache, budget, fingerprint)?
            {
                return Ok(Some(output));
            }
            output
        };
        // Only publish entries after the request output has been assembled.
        if !inserts.is_empty() {
            cache.insert_batch(inserts);
        }
        Ok(Some(output))
    })
    .await
    .context("batch image preprocessing task panicked")?
}

/// Decode and preprocess an image batch.
fn preprocess_image_frames(
    images: &[Arc<ImageFrame>],
    vision_processors: &VisionProcessors,
    image_wire_dtype: Option<EncoderDtype>,
) -> Result<PreprocessedEncoderInputs> {
    let decoded = decode_image_frames(images)?;
    let processor = vision_processors.image()?;
    preprocess_images(processor.as_ref(), &decoded, image_wire_dtype)
}

/// Decode the frames in parallel and borrow them for the processor.
fn decode_image_frames(images: &[Arc<ImageFrame>]) -> Result<Vec<&image::DynamicImage>> {
    ImageFrame::decode_all(images).map_err(llm_multimodal::MultiModalError::Media)?;
    images
        .iter()
        .map(|frame| {
            frame
                .image()
                .map_err(llm_multimodal::MultiModalError::Media)
                .map_err(anyhow::Error::from)
        })
        .collect()
}

/// Admit a distinct batch's image views without making a second pixel allocation.
fn admit_shared_batch(
    batch: &mut PreprocessedEncoderInputs,
    images: &[Arc<ImageFrame>],
    layouts: &EncoderFieldLayouts,
    cache: &PixelCache,
    budget: usize,
    fingerprint: u64,
) -> Result<bool> {
    if !batch.encoder_input.share_wire(budget) {
        return Ok(false);
    }
    let items = split_batch(batch, layouts, images.len())?;
    cache.insert_batch(images.iter().zip(items).map(|(image, preprocessed)| {
        let key = PixelCacheKey {
            image_hash: image.hash.clone(),
            config_fingerprint: fingerprint,
        };
        (key, Arc::new(CachedPreprocessedItem { preprocessed }))
    }));
    Ok(true)
}

/// Slice each image's patches and metadata, sharing already immutable wire data.
fn split_batch(
    batch: &PreprocessedEncoderInputs,
    layouts: &EncoderFieldLayouts,
    image_count: usize,
) -> Result<Vec<PreprocessedEncoderInputs>> {
    let sizes = match &layouts.encoder_input {
        FieldLayout::Batched => vec![1; image_count],
        FieldLayout::Flat { sizes_key } => batch
            .model_specific
            .get(sizes_key)
            .context("missing image patch counts")?
            .as_flat_sizes()?,
    };
    anyhow::ensure!(
        sizes.len() == image_count
            && batch.item_sizes.len() == image_count
            && batch.feature_token_counts.len() == image_count,
        "image preprocessing returned an unexpected item count"
    );
    let inputs = batch.encoder_input.split_axis0(&sizes)?;
    let mut items = Vec::with_capacity(image_count);
    for (index, encoder_input) in inputs.into_iter().enumerate() {
        let model_specific = batch
            .model_specific
            .iter()
            .map(|(key, value)| {
                let value = match layouts.model_specific.get(key) {
                    Some(FieldLayout::Batched) => value.slice_first_dim(index, 1)?,
                    Some(FieldLayout::Flat { .. }) => {
                        anyhow::bail!("unexpected flat image cache metadata {key}")
                    }
                    None => value.clone(),
                };
                Ok((key.clone(), value))
            })
            .collect::<Result<_>>()?;
        items.push(PreprocessedEncoderInputs {
            encoder_input,
            model_specific,
            feature_token_counts: vec![batch.feature_token_counts[index]],
            item_sizes: vec![batch.item_sizes[index]],
        });
    }
    Ok(items)
}
