//! Cache-aware image preprocessing: deduplicate, process misses, assemble, publish.

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use llm_multimodal::{
    EncoderFieldLayouts, ImageFrame, PreProcessorConfig, PreprocessedEncoderInputs,
    VisionProcessorRegistry,
};
use tracing::warn;

use super::{
    config::{MultimodalComponents, MultimodalModelConfig},
    pixel_cache::{config_fingerprint, CachedPreprocessedItem, PixelCacheKey},
};
use crate::observability::metrics::Metrics;

/// Cache entries own independent per-image buffers. Hits and duplicate positions
/// share Arc handles; only final assembly copies their pixels. All unique misses
/// use owned per-image outputs when supported and assembly is required;
/// otherwise they are processed in one batch.
pub(super) async fn preprocess_images_cached_batch(
    images: &[Arc<ImageFrame>],
    components: &MultimodalComponents,
    model_id: &str,
    model_type: Option<&str>,
    tokenizer_id: &str,
    model_config: &MultimodalModelConfig,
    layouts: &EncoderFieldLayouts,
) -> Result<Option<PreprocessedEncoderInputs>> {
    let pp_config = &model_config.preprocessor_config;
    let cache = components
        .pixel_cache
        .clone()
        .context("pixel cache is not configured")?;
    let mut config = serde_json::json!({
        "model_id": model_id, "model_type": model_type,
        "model_config": model_config.config, "preprocessor_config": pp_config,
    });
    config.sort_all_objects();
    let fingerprint = config_fingerprint(tokenizer_id, &config);

    let mut unique = HashMap::new();
    let mut order = Vec::with_capacity(images.len());
    let mut parts = Vec::new();
    let mut misses = Vec::new();
    let mut miss_images = Vec::new();
    for image in images {
        let key = PixelCacheKey {
            image_hash: image.hash.clone(),
            config_fingerprint: fingerprint,
        };
        let next = unique.len();
        let index = *unique.entry(key.clone()).or_insert(next);
        order.push(index);
        if index != next {
            continue;
        }
        parts.push(cache.get(&key));
        Metrics::record_mm_pixel_cache_lookup(model_id, parts[index].is_some());
        if parts[index].is_none() {
            misses.push((index, key));
            miss_images.push(image.clone());
        }
    }

    let miss_outputs = if miss_images.is_empty() {
        ImageCacheMissOutputs::Items(Vec::new())
    } else {
        preprocess_image_cache_misses(
            components.vision_processor_registry.clone(),
            model_id.to_string(),
            model_type.map(String::from),
            pp_config.clone(),
            miss_images,
            misses.len() != images.len(),
        )
        .await?
    };

    let layouts = layouts.clone();
    let image_count = images.len();

    let result = tokio::task::spawn_blocking(move || -> Result<PreprocessedEncoderInputs> {
        let (batch, items) = match miss_outputs {
            ImageCacheMissOutputs::Batch(batch) => match batch.slice_items(&layouts) {
                Ok(items) => (Some(*batch), items),
                Err(error) if misses.len() == image_count => {
                    warn!(
                        error = format!("{error:#}"),
                        image_count,
                        "Cannot split cold image batch for caching; returning it uncached"
                    );
                    return Ok(*batch);
                }
                Err(error) => return Err(error),
            },
            ImageCacheMissOutputs::Items(items) => (None, items),
        };
        anyhow::ensure!(
            items.len() == misses.len(),
            "miss batch item count mismatch"
        );
        let mut new_entries = Vec::with_capacity(misses.len());
        for (preprocessed, (index, key)) in items.into_iter().zip(misses) {
            anyhow::ensure!(
                preprocessed.feature_token_counts.len() == 1 && preprocessed.item_sizes.len() == 1,
                "cache entry must describe exactly one image"
            );
            let entry = Arc::new(CachedPreprocessedItem { preprocessed });
            parts[index] = Some(entry.clone());
            new_entries.push((key, entry));
        }

        let direct_batch = batch.is_some() && new_entries.len() == image_count;
        // With no hits or duplicate positions, the miss batch already is the
        // final output. Return it directly instead of allocating another copy.
        let output = if direct_batch {
            batch.context("missing cold batch")?
        } else {
            // Item copies now own the miss pixels. Release the temporary miss
            // batch before allocating the final mixed/duplicate output.
            drop(batch);
            let references = order
                .iter()
                .map(|&index| {
                    parts[index]
                        .as_ref()
                        .map(|entry| &entry.preprocessed)
                        .context("missing image output")
                })
                .collect::<Result<Vec<_>>>()?;
            PreprocessedEncoderInputs::concat_refs(&references, &layouts.model_specific)?
        };

        // Publish only fully validated entries. Eviction never invalidates the
        // Arc handles held by another in-flight request.
        for (key, entry) in new_entries {
            cache.insert(key, entry);
        }
        Ok(output)
    })
    .await
    .context("Cached image batch assembly task panicked")?;
    match result {
        Ok(output) => Ok(Some(output)),
        Err(error) => {
            warn!(error = format!("{error:#}"), image_count,
                "Cached image outputs could not be assembled; falling back to whole-batch preprocessing");
            Ok(None)
        }
    }
}

enum ImageCacheMissOutputs {
    Batch(Box<PreprocessedEncoderInputs>),
    Items(Vec<PreprocessedEncoderInputs>),
}

async fn preprocess_image_cache_misses(
    registry: Arc<VisionProcessorRegistry>,
    model_id: String,
    model_type: Option<String>,
    pp_config: PreProcessorConfig,
    images: Vec<Arc<ImageFrame>>,
    prefer_items: bool,
) -> Result<ImageCacheMissOutputs> {
    tokio::task::spawn_blocking(move || {
        ImageFrame::decode_all(&images).map_err(llm_multimodal::MultiModalError::Media)?;
        let processor = registry
            .find(&model_id, model_type.as_deref())
            .ok_or_else(|| anyhow::anyhow!("No vision processor found for model: {model_id}"))?;
        if prefer_items {
            let references = images
                .iter()
                .map(|frame| {
                    frame
                        .image()
                        .map_err(llm_multimodal::MultiModalError::Media)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(result) = processor.preprocess_image_items(&references, &pp_config) {
                return result
                    .map(ImageCacheMissOutputs::Items)
                    .map_err(|e| anyhow::anyhow!("Image preprocessing failed: {e}"));
            }
        }
        let raw_images = images
            .iter()
            .map(|frame| {
                frame
                    .image()
                    .cloned()
                    .map_err(llm_multimodal::MultiModalError::Media)
            })
            .collect::<Result<Vec<_>, _>>()?;
        processor
            .preprocess(&raw_images, &pp_config)
            .map(|batch| ImageCacheMissOutputs::Batch(Box::new(batch)))
            .map_err(|e| anyhow::anyhow!("Image preprocessing failed: {e}"))
    })
    .await
    .context("Preprocessing task panicked")?
}
