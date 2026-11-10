//! Cache MiniMax M3 images independently while preprocessing misses as a batch.

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use llm_multimodal::{
    EncoderDtype, EncoderFieldLayouts, EncoderInput, FieldLayout, ImageFrame, Modality,
    PreprocessedEncoderInputs,
};

use super::{
    decode_image_frames, on_pool, preprocess_image_frames, preprocess_images, PlanContext,
};
use crate::{
    observability::metrics::Metrics,
    routers::grpc::multimodal::pixel_cache::{CachedPreprocessedItem, PixelCache, PixelCacheKey},
};

/// Reuse individual images, batch the unique misses, and restore request order.
pub(super) async fn preprocess(
    images: &[Arc<ImageFrame>],
    cache: &Arc<PixelCache>,
    ctx: &PlanContext<'_>,
    fingerprint: u64,
) -> Result<PreprocessedEncoderInputs> {
    let images = images.to_vec();
    let cache = Arc::clone(cache);
    let model_id = Metrics::interned_model_label(ctx.model_id);
    let processors = Arc::clone(&ctx.vision_processors);
    let wire_dtype = ctx.image_wire_dtype;
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

        let budget = cache.max_bytes();
        let storage_width = if wire_dtype == Some(EncoderDtype::BFloat16) {
            2
        } else {
            1
        };
        // An oversized batch of distinct images cannot fill the cache. Keep
        // the contiguous batch path before looking up or splitting any items.
        // Repeated images can still benefit from request-local deduplication.
        if unique.len() == images.len() {
            let processor = processors.image()?;
            let over_budget = unique
                .iter()
                .filter_map(|image| {
                    let size = image.size();
                    processor.image_input_elements(size.width, size.height)
                })
                .try_fold(budget / storage_width, usize::checked_sub)
                .is_none();
            if over_budget {
                return preprocess_image_frames(&images, &processors, wire_dtype);
            }
        }

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

        let mut inserts = Vec::new();
        let mut cold_batch = None;
        if !misses.is_empty() {
            let direct = misses.len() == images.len();
            let processor = processors.image()?;
            let decoded = decode_image_frames(&miss_images)?;
            let items = if let Some(items) = (!direct)
                .then(|| processor.preprocess_image_items(&decoded, wire_dtype))
                .flatten()
            {
                items?
            } else {
                let mut batch = preprocess_images(processor.as_ref(), &decoded, wire_dtype)?;
                // The fused batch processor emits neutral pixels for a lone
                // image. Match the representation of multi-image entries.
                if wire_dtype == Some(EncoderDtype::BFloat16)
                    && !matches!(batch.encoder_input, EncoderInput::Wire { .. })
                {
                    let input = batch.encoder_input;
                    let shape = input.shape().to_vec();
                    let norm = input
                        .pixel_norm()
                        .context("M3 image input has no pixel normalization")?;
                    let bytes = input
                        .into_bytes_as(EncoderDtype::BFloat16)
                        .map_err(anyhow::Error::msg)?;
                    batch.encoder_input =
                        EncoderInput::wire(bytes, EncoderDtype::BFloat16, shape, norm)
                            .map_err(anyhow::Error::msg)?;
                }
                if direct && batch.encoder_input.nbytes() > budget {
                    return Ok(batch);
                }
                let items = split_batch(&batch, &layouts, misses.len())?;
                if direct {
                    cold_batch = Some(batch);
                }
                items
            };
            anyhow::ensure!(
                items.len() == misses.len()
                    && items
                        .iter()
                        .all(|item| item.batch_size() == 1 && item.feature_token_counts.len() == 1),
                "M3 image preprocessing returned an unexpected item count"
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

        let output = if let Some(batch) = cold_batch {
            batch
        } else {
            let ordered = order
                .into_iter()
                .map(|index| {
                    parts[index]
                        .as_ref()
                        .map(|entry| &entry.preprocessed)
                        .context("missing M3 image cache output")
                })
                .collect::<Result<Vec<_>>>()?;
            PreprocessedEncoderInputs::concat_refs(&ordered, &layouts.model_specific)?
        };
        // Only publish entries after the request output has been assembled.
        for (key, entry) in inserts {
            cache.insert(key, entry);
        }
        Ok(output)
    })
    .await
    .context("M3 image preprocessing task panicked")?
}

/// Own each image's patches and metadata so the LRU can evict images independently.
fn split_batch(
    batch: &PreprocessedEncoderInputs,
    layouts: &EncoderFieldLayouts,
    image_count: usize,
) -> Result<Vec<PreprocessedEncoderInputs>> {
    let FieldLayout::Flat { sizes_key } = &layouts.encoder_input else {
        anyhow::bail!("expected a flat M3 image encoder input");
    };
    let sizes = batch
        .model_specific
        .get(sizes_key)
        .context("missing M3 image patch counts")?
        .as_flat_sizes()?;
    anyhow::ensure!(
        sizes.len() == image_count
            && batch.item_sizes.len() == image_count
            && batch.feature_token_counts.len() == image_count,
        "M3 image preprocessing returned an unexpected item count"
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
                        anyhow::bail!("unexpected flat M3 image metadata {key}")
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
