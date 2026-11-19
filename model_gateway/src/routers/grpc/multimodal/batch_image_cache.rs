//! Cache images independently while preprocessing misses as a batch.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{Context, Result};
use llm_multimodal::{
    buffer_pool, EncoderDtype, EncoderFieldLayouts, EncoderInput, FieldLayout, ImageFrame,
    Modality, PreprocessedEncoderInputs,
};

use super::{image_cache_fingerprint, on_pool, preprocess_image_frames, PlanContext};
use crate::{
    observability::metrics::Metrics,
    routers::grpc::multimodal::pixel_cache::{CachedPreprocessedItem, PixelCache, PixelCacheKey},
};

/// Reuse individual images, batch the unique misses, and restore request order.
pub(super) async fn preprocess(
    images: &[Arc<ImageFrame>],
    cache: &Arc<PixelCache>,
    ctx: &PlanContext<'_>,
    newly_cacheable: bool,
) -> Result<Option<PreprocessedEncoderInputs>> {
    let model_id = Metrics::interned_model_label(ctx.model_id);
    let wire_dtype = Some(EncoderDtype::BFloat16);
    let budget = cache.max_bytes();
    let width = EncoderDtype::BFloat16.element_size();
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
    let working_capacity = buffer_pool::allocation_capacity(working_bytes);
    let mut indices = HashMap::new();
    let mut order = Vec::with_capacity(images.len());
    let mut parts = Vec::new();
    let mut misses = Vec::new();
    let mut miss_images = Vec::new();
    for image in images {
        let next = parts.len();
        let index = *indices.entry(image.hash.as_str()).or_insert(next);
        order.push(index);
        if index != next {
            continue;
        }
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

    const LARGE_WIRE_BYTES: usize = 8 * 1024 * 1024;
    let direct = misses.len() == images.len();
    // Leave large pooled buffers available to other cold waves until the
    // incoming images recur. The caller handles this with its normal batch.
    if direct
        && newly_cacheable
        && (LARGE_WIRE_BYTES..=buffer_pool::LARGEST).contains(&working_bytes)
        && !cache.admit_repeated(misses.iter().map(|(_, key)| key), working_capacity)
    {
        return Ok(None);
    }

    let distinct = parts.len() == images.len();
    let has_misses = !misses.is_empty();
    let images = images.to_vec();
    let cache = Arc::clone(cache);
    let processors = Arc::clone(&ctx.vision_processors);
    let layouts = ctx.spec.encoder_field_layouts_for(Modality::Image);
    on_pool(ctx.model_id, move || {
        let mut inserts = Vec::new();
        let mut direct_output = None;
        if has_misses {
            // For a large, mostly new request, keep one contiguous output.
            // Recompute the small set of hits rather than gather a second
            // full tensor. Keep the hit-assisted path if decoding fails.
            if distinct
                && !direct
                && misses.len() >= images.len() - misses.len()
                && working_bytes >= LARGE_WIRE_BYTES
                && working_capacity < budget
            {
                if let Ok(mut batch) = preprocess_image_frames(&images, &processors, wire_dtype) {
                    if !admit_shared_batch(&mut batch, &images, &layouts, &cache, fingerprint)?
                        && batch.encoder_input.nbytes() <= budget
                        && batch.encoder_input.share_wire(usize::MAX)
                    {
                        // Keep the old owners. Copy only the recomputed misses
                        // into compact entries so they do not pin the full batch.
                        let mut misses = misses.into_iter().peekable();
                        let inserts = split_batch(&batch, &layouts, images.len())?
                            .into_iter()
                            .enumerate()
                            .filter_map(|(index, mut preprocessed)| {
                                let (_, key) = misses.next_if(|(miss, _)| *miss == index)?;
                                preprocessed.encoder_input =
                                    preprocessed.encoder_input.view().to_owned();
                                Some((key, Arc::new(CachedPreprocessedItem { preprocessed })))
                            })
                            .collect::<Vec<_>>();
                        cache.insert_batch(inserts);
                    }
                    return Ok(Some(batch));
                }
            }
            let mut batch = preprocess_image_frames(&miss_images, &processors, wire_dtype)?;
            // MiniMax M3 uses neutral pixels for a single image. Store the
            // BF16 form so the entry can join a multi-image wire batch.
            if !matches!(batch.encoder_input, EncoderInput::Wire { .. }) {
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
            // An allocation that fills the budget leaves no room for entries.
            // Let the owned split copy those slices in parallel.
            batch.encoder_input.share_wire(budget.saturating_sub(1));
            let mut items = split_batch(&batch, &layouts, misses.len())?;
            // When pooled capacity leaves insufficient room for metadata,
            // compact the cache entries while retaining the request output.
            if let Some((_, capacity)) = batch.encoder_input.shared_allocation() {
                if !cache.shared_batch_fits(capacity, items.iter()) {
                    for item in &mut items {
                        item.encoder_input = item.encoder_input.view().to_owned();
                    }
                }
            }
            if direct {
                direct_output = Some(batch);
            }
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
            if distinct
                && has_misses
                && admit_shared_batch(&mut output, &images, &layouts, &cache, fingerprint)?
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

/// Admit a distinct batch's image views without making a second pixel allocation.
fn admit_shared_batch(
    batch: &mut PreprocessedEncoderInputs,
    images: &[Arc<ImageFrame>],
    layouts: &EncoderFieldLayouts,
    cache: &PixelCache,
    fingerprint: u64,
) -> Result<bool> {
    if !batch.encoder_input.share_wire(cache.max_bytes()) {
        return Ok(false);
    }
    let (_, capacity) = batch
        .encoder_input
        .shared_allocation()
        .context("missing shared wire capacity")?;
    let items = split_batch(batch, layouts, images.len())?;
    let entries = images
        .iter()
        .zip(items)
        .map(|(image, preprocessed)| {
            let key = PixelCacheKey {
                image_hash: image.hash.clone(),
                config_fingerprint: fingerprint,
            };
            (key, Arc::new(CachedPreprocessedItem { preprocessed }))
        })
        .collect();
    Ok(cache.insert_shared_batch(entries, capacity))
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
